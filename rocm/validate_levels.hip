/**
 * Correctness test: compare any optimization level against the level 1 reference.
 *
 * Usage:
 *   ./validate_levels [-l <level>] [-K <k>] [-N <nfuncs>]
 *
 *   -l  level to validate (2-7, default 3)
 *         2  L2: B cached in LDS
 *         3  L3: register blocking (K-templated)
 *         4  L4: AMD MFMA + L3 fallback
 *         5  L5: wave-specialised double-buffering MFMA (HIP only)
 *         6  L6: Kronecker product GEMM (hipBLAS)
 *         7  L7: multi-wave MFMA, B resident in VGPRs (HIP only)
 *   -K  single K value; if omitted sweeps K in {4,6,8,10}
 *   -N  batch size (default 16)
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "util.h"
#include "transform.h"          // L1 — reference
#include "transform_level2.h"   // L2
#include "transform_level3.h"   // L3
#include "transform_level4.h"   // L4
#ifdef MRA_HAVE_HIP
#  include "transform_level5.h"   // L5 — wave-specialised MFMA (HIP only)
#  include "transform_level7.h"   // L7 — multi-wave MFMA, B in VGPRs (HIP only)
#else
#  include "transform_cublasdx.h" // L5 — cuBLASDx (NVIDIA only)
#endif
#include "transform_rocwmma.h"
#include "transform_kron.h"     // L6

template <typename T>
void test_level(int level, int K, int nfuncs) {
    const int K3     = K * K * K;
    const int nblocks = nfuncs;

    // Allocate and fill host arrays with random data
    std::vector<T> h_A(nfuncs * K3), h_B(K * K);
    std::vector<T> h_Cref(nfuncs * K3), h_Ctest(nfuncs * K3);
    std::srand(42);
    for (auto& v : h_A) v = (T)std::rand() / RAND_MAX;
    for (auto& v : h_B) v = (T)std::rand() / RAND_MAX;

    // Device allocations
    T *d_A, *d_B, *d_Cref, *d_Ctest, *d_workspace_ref, *d_workspace;
    MALLOC(&d_A,            nfuncs * K3 * sizeof(T));
    MALLOC(&d_B,            K * K * sizeof(T));
    MALLOC(&d_Cref,         nfuncs * K3 * sizeof(T));
    MALLOC(&d_Ctest,        nfuncs * K3 * sizeof(T));
    MALLOC(&d_workspace_ref, nfuncs * K3 * sizeof(T));
    MALLOC(&d_workspace,    nfuncs * K3 * sizeof(T));

    T *d_KronMat = nullptr;
    if (level == 8) {
        MALLOC(&d_KronMat, (size_t)K3 * K3 * sizeof(T));
    }

    // Copy inputs to device
    MEMCPY_H2D(d_A, h_A.data(), nfuncs * K3 * sizeof(T));
    MEMCPY_H2D(d_B, h_B.data(), K * K * sizeof(T));

    Stream stream;
    CREATE_STREAM(&stream);

    // --- Reference: level 1 ---
    submit_transform_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Cref, d_workspace_ref, stream);
    SYNC_STREAM(stream);

    // --- Tested level ---
    switch (level) {
        case 2:
            submit_transform_level2_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 3:
            submit_transform_level3_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 4:
            submit_transform_level4_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 5:
            submit_transform_level5_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 6:
            submit_transform_rocwmma_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 7:
            submit_transform_level7_bench<T>(nfuncs, nblocks, K, d_A, d_B, d_Ctest, d_workspace, stream);
            SYNC_STREAM(stream);
            break;
        case 8: {
            build_kron_matrix<T>(K, d_B, d_KronMat, stream);
            SYNC_STREAM(stream);
            blasHandle_t blas_handle;
            blasCreate(&blas_handle);
            submit_transform_kron_bench<T>(nfuncs, K, d_A, d_KronMat, d_Ctest, blas_handle, stream);
            SYNC_STREAM(stream);
            blasDestroy(blas_handle);
            break;
        }
        default:
            std::cerr << "Unknown level " << level << " (valid: 2-7)\n";
            FREE(d_A); FREE(d_B); FREE(d_Cref); FREE(d_Ctest);
            FREE(d_workspace_ref); FREE(d_workspace);
            if (d_KronMat) FREE(d_KronMat);
#ifdef MRA_HAVE_HIP
            (void)hipStreamDestroy(stream);
#else
            (void)cudaStreamDestroy(stream);
#endif
            return;
    }

    // Copy results to host
    MEMCPY_D2H(h_Cref.data(),  d_Cref,  nfuncs * K3 * sizeof(T));
    MEMCPY_D2H(h_Ctest.data(), d_Ctest, nfuncs * K3 * sizeof(T));

    // Compare
    T max_abs_err = 0, max_rel_err = 0;
    for (int i = 0; i < nfuncs * K3; ++i) {
        T abs_err = std::abs(h_Cref[i] - h_Ctest[i]);
        T rel_err = abs_err / (std::abs(h_Cref[i]) + 1e-14);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    if (max_rel_err >= 1e-10) {
        std::cout << "FAIL!\n";
        for (int i = 0; i < nfuncs * K3; ++i) {
            if (i % K3 == 0) {
                std::cout << "  Function " << i / K3 << ":\n";
            }
            int idx_k = (i % K3) / (K * K);
            int idx_j = (i % (K * K)) / K;
            int idx_i = i % K;
            std::cout << "    [" << idx_k << "][" << idx_j << "][" << idx_i << "]  ref="
                      << h_Cref[i] << "  test=" << h_Ctest[i] << "\n";
        }
    }

    std::cout << "K=" << K << " nfuncs=" << nfuncs << " level=" << level
              << "  max_abs_err=" << max_abs_err
              << "  max_rel_err=" << max_rel_err
              << (max_rel_err < 1e-10 ? "  PASS" : "  FAIL")
              << "\n";

    FREE(d_A); FREE(d_B); FREE(d_Cref); FREE(d_Ctest);
    FREE(d_workspace_ref); FREE(d_workspace);
    if (d_KronMat) FREE(d_KronMat);
#ifdef MRA_HAVE_HIP
    (void)hipStreamDestroy(stream);
#else
    (void)cudaStreamDestroy(stream);
#endif
}

int main(int argc, char** argv) {
    OptionParser opts(argc, argv);
    int level  = opts.parse(std::string("-l"), 3);
    int nfuncs = opts.parse(std::string("-N"), 16);

    if (opts.exists(std::string("-K"))) {
        int K = opts.parse(std::string("-K"), 8);
        test_level<double>(level, K, nfuncs);
    } else {
        for (int K : {4, 6, 8, 10}) {
            test_level<double>(level, K, nfuncs);
        }
    }
    return 0;
}
