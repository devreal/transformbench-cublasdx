/**
 * validate.hip — compare GPU L1 output against the CPU reference transform.
 *
 * CPU reference mirrors madness/src/madness/tensor/transform3d.cc :: transform3d<T>():
 *
 *   result(a,b,c) = sum(i',j',k') A(i',j',k') C(i',a) C(j',b) C(k',c)
 *
 * implemented as three sequential mTxm passes:
 *   mTxm(K², K, K, r,   A, B)    pass 1: contract dim-0
 *   mTxm(K², K, K, tmp, r, B)    pass 2: contract dim-1
 *   mTxm(K², K, K, r, tmp, B)    pass 3: contract dim-2
 *
 * where  mTxm(dimi, dimj, dimk, c, a, b):
 *   c(i,j) += sum(k)  a(k,i) * b(k,j)
 */

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "transform.h"
#include "util.h"

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

// c(i,j) += sum(k) a(k,i) * b(k,j)   (from MADNESS mTxm)
static void cpu_mTxm(int dimi, int dimj, int dimk,
                     double* c, const double* a, const double* b) {
    for (int i = 0; i < dimi; ++i) {
        double* ci         = c + i * dimj;
        const double* aik  = a + i;          // a(k,i) strides by dimi
        for (int k = 0; k < dimk; ++k, aik += dimi) {
            double aki = *aik;
            for (int j = 0; j < dimj; ++j)
                ci[j] += aki * b[k * dimj + j];
        }
    }
}

// Three-pass transform matching transform3d.cc
static void cpu_transform3d(int K, const double* A, const double* B, double* result) {
    int K3 = K * K * K;
    std::vector<double> tmp(K3, 0.0);
    std::fill(result, result + K3, 0.0);

    cpu_mTxm(K * K, K, K, result,    A,           B);  // pass 1
    cpu_mTxm(K * K, K, K, tmp.data(), result,     B);  // pass 2
    std::fill(result, result + K3, 0.0);
    cpu_mTxm(K * K, K, K, result,    tmp.data(),  B);  // pass 3
}

// ---------------------------------------------------------------------------
// Block-distributed transform (corner-turn variant).
//
// Block A along the fastest tensor index k':  block s owns A[:,:,k'=s] as a
// K x K matrix indexed by (i', j').
//
//   Pass 1 (contract i'):   local.   blk_s <- B^T . blk_s        -> [a, j']
//   Pass 2 (contract j'):   local.   blk_s <- blk_s . B          -> [a, b]
//   Pass 3 (contract k'):   all-to-all "corner turn" where block t pulls row
//                           a=t from every other block, storing into the
//                           (b, k') layout.  Then local right-multiply:
//                                     blk_t <- blk_t . B         -> [b, c]
//
// Applying B on the right in pass 3 makes the output land in canonical (b, c)
// order directly; the in-block transpose is absorbed into the corner turn for
// free (same number of loads/stores, different destination cell).
// ---------------------------------------------------------------------------

static void print_blocks(const char* label, int K,
                         const std::vector<std::vector<double>>& blk) {
    std::cout << "  " << label << ":\n";
    for (int s = 0; s < (int)blk.size(); ++s) {
        std::cout << "    block " << s << ":\n";
        for (int i = 0; i < K; ++i) {
            std::cout << "     ";
            for (int j = 0; j < K; ++j)
                std::cout << " " << std::setw(8) << blk[s][i*K + j];
            std::cout << "\n";
        }
    }
}

static void cpu_transform3d_blocked(int K, const double* A, const double* B,
                                    double* out, bool trace = false) {
    const int K2 = K * K;
    std::vector<std::vector<double>> blk(K, std::vector<double>(K2));

    // distribute: block s := A[:,:,k=s]  (strided read from the flat layout)
    for (int s = 0; s < K; ++s)
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                blk[s][i*K + j] = A[i*K2 + j*K + s];
    if (trace) print_blocks("after distribute (block_s[i,j] = A[i,j,s])", K, blk);

    // Pass 1: contract i'.  blk_s <- B^T . blk_s       -> indexed [a, j]
    for (int s = 0; s < K; ++s) {
        std::vector<double> t(K2, 0.0);
        for (int a = 0; a < K; ++a)
            for (int j = 0; j < K; ++j)
                for (int i = 0; i < K; ++i)
                    t[a*K + j] += B[i*K + a] * blk[s][i*K + j];
        blk[s] = std::move(t);
    }
    if (trace) print_blocks("after pass 1 (block_s[a,j])", K, blk);

    // Pass 2: contract j'.  blk_s <- blk_s . B         -> indexed [a, b]
    for (int s = 0; s < K; ++s) {
        std::vector<double> t(K2, 0.0);
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < K; ++b)
                for (int j = 0; j < K; ++j)
                    t[a*K + b] += blk[s][a*K + j] * B[j*K + b];
        blk[s] = std::move(t);
    }
    if (trace) print_blocks("after pass 2 (block_s[a,b], fixed k'=s)", K, blk);

    // Corner turn: block t pulls row a=t from every block, stored as (b, k').
    // Before: blk[s][a, b]    (block-index = k' = s)
    // After : blk[t][b, k]    (block-index = a = t)
    {
        std::vector<std::vector<double>> exch(K, std::vector<double>(K2));
        for (int t = 0; t < K; ++t)
            for (int b = 0; b < K; ++b)
                for (int k = 0; k < K; ++k)
                    exch[t][b*K + k] = blk[k][t*K + b];
        blk = std::move(exch);
    }
    if (trace) print_blocks("after corner turn (block_t[b,k], fixed a=t)", K, blk);

    // Pass 3: contract k'.  blk_t <- blk_t . B         -> indexed [b, c]
    for (int t = 0; t < K; ++t) {
        std::vector<double> r(K2, 0.0);
        for (int b = 0; b < K; ++b)
            for (int c = 0; c < K; ++c)
                for (int k = 0; k < K; ++k)
                    r[b*K + c] += blk[t][b*K + k] * B[k*K + c];
        blk[t] = std::move(r);
    }
    if (trace) print_blocks("after pass 3 (block_t[b,c] = result[t,b,c])", K, blk);

    // Canonical write-out: block_a holds result[a, :, :] in (b, c) order.
    for (int a = 0; a < K; ++a)
        for (int b = 0; b < K; ++b)
            for (int c = 0; c < K; ++c)
                out[a*K2 + b*K + c] = blk[a][b*K + c];
}

// K=2 walkthrough with the counting tensor A[i,j,k] = 100i + 10j + k and B = I.
// With B = I the GEMMs become no-ops, so every stage just shows the current
// distribution of elements across blocks -- the shuffle pattern is laid bare.
// Expected final output == A (since B = I acts as identity in all three passes).
static void debug_blocked_K2() {
    const int K = 2, K2 = 4, K3 = 8;

    std::vector<double> A(K3);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            for (int k = 0; k < K; ++k)
                A[i*K2 + j*K + k] = 100.0*i + 10.0*j + k;

    std::vector<double> B(K2, 0.0);
    for (int i = 0; i < K; ++i) B[i*K + i] = 1.0;

    std::cout << "=== debug_blocked_K2 (A[i,j,k] = 100i+10j+k, B = I) ===\n";
    std::cout << "  B = I makes every GEMM a no-op; the only motion is the shuffle.\n";
    std::cout << "  expected final output == A.\n";

    std::vector<double> out(K3, 0.0);
    cpu_transform3d_blocked(K, A.data(), B.data(), out.data(), /*trace=*/true);

    double max_err = 0.0;
    for (int i = 0; i < K3; ++i)
        max_err = std::max(max_err, std::abs(out[i] - A[i]));
    std::cout << "  final max_abs_err vs A = " << max_err
              << (max_err < 1e-10 ? "  MATCH" : "  DIFFER") << "\n\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto opt = OptionParser(argc, argv);
    int K = opt.parse("-K", 16);
    int N = opt.parse("-N", 8);
    int debug = opt.parse("-debug", 0);

    if (debug) {
        debug_blocked_K2();
        return 0;
    }

    const int K3 = K * K * K;

    // Deterministic input
    std::vector<double> h_A(N * K3);
    std::vector<double> h_B(K * K);
    for (int i = 0; i < N * K3; ++i) h_A[i] = (double)(i % 13) * 0.1 - 0.6;
    for (int i = 0; i < K * K;  ++i) h_B[i] = (double)(i % 7)  * 0.1 - 0.3;

    // CPU reference (3D tensor transform)
    std::vector<double> h_ref(N * K3);
    for (int n = 0; n < N; ++n)
        cpu_transform3d(K, h_A.data() + n * K3, h_B.data(), h_ref.data() + n * K3);

    // Blocked algorithm: per-tensor distribute along k', corner-turn for pass 3
    std::vector<double> h_blk(N * K3, 0.0);
    for (int n = 0; n < N; ++n)
        cpu_transform3d_blocked(K, h_A.data() + n * K3, h_B.data(), h_blk.data() + n * K3);

    double blk_max_err = 0.0;
    for (int i = 0; i < N * K3; ++i)
        blk_max_err = std::max(blk_max_err, std::abs(h_blk[i] - h_ref[i]));
    std::cout << "K=" << K << " N=" << N
              << " blk_vs_ref max_abs_err=" << blk_max_err
              << (blk_max_err < 1e-10 ? "  MATCH" : "  DIFFER")
              << std::endl;

    // GPU L1
    double *d_A, *d_B, *d_C, *d_workspace;
    MALLOC(&d_A,         N * K3    * sizeof(double));
    MALLOC(&d_B,             K * K * sizeof(double));
    MALLOC(&d_C,         N * K3    * sizeof(double));
    MALLOC(&d_workspace,     K3    * sizeof(double));  // 1 block

    MEMCPY_H2D(d_A, h_A.data(), N * K3    * sizeof(double));
    MEMCPY_H2D(d_B, h_B.data(),     K * K * sizeof(double));

    Stream stream;
    CREATE_STREAM(&stream);
    submit_transform_bench<double>(N, /*nblocks=*/1, K, d_A, d_B, d_C, d_workspace, stream);
    SYNC_STREAM(stream);

    std::vector<double> h_gpu(N * K3);
    MEMCPY_D2H(h_gpu.data(), d_C, N * K3 * sizeof(double));

    FREE(d_A); FREE(d_B); FREE(d_C); FREE(d_workspace);

    // Compare
    double max_abs_err = 0.0, max_ref_val = 0.0;
    for (int i = 0; i < N * K3; ++i) {
        double err = std::abs(h_gpu[i] - h_ref[i]);
        double ref = std::abs(h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (ref > max_ref_val) max_ref_val = ref;
    }
    double rel_err = (max_ref_val > 0.0) ? max_abs_err / max_ref_val : max_abs_err;

    bool pass = rel_err < 1e-10;
    std::cout << "K=" << K
              << " N=" << N
              << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << rel_err
              << (pass ? "  PASS" : "  FAIL")
              << std::endl;

    return pass ? 0 : 1;
}
