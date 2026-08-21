# transformbench-cublasdx

Benchmark suite for the **MRA (Multi-Resolution Analysis) transform operator** — a batched 3D tensor-times-matrix contraction — targeting **AMD MI250X** (HIP/ROCm) and **NVIDIA** (CUDA/cuBLASDx) GPUs.

## Mathematical Core

The transform applies a K×K matrix B (transposed) along each dimension of a K×K×K tensor:

```
for d in {0, 1, 2}:
    C ← B^T × A    (in-place, cycling through a workspace)

GEMM shape per pass:
  A  (input):   K² × K   col-major
  B  (matrix):  K  × K   row-major
  C  (output):  K² × K   row-major
  FLOPs: 2 × K² × K × K  per pass  →  3 × 2 × K⁴  per full transform
```

## Optimization Levels

| Level | File | Technique | Smem | Threads | Notes |
|---|---|---|---|---|---|
| L1 | `mxm.h` / `transform.h` | Global memory only | 0 | 128×1×1 | Reference; A reloaded K× per row |
| L2 | `mxm_level2.h` / `transform_level2.h` | B cached in LDS | K²·8 B | 128×1×1 | Eliminates B HBM redundancy |
| L3 | `mxm_level3.h` / `transform_level3.h` | Register blocking (K-templated) | K²·8 B | 128×1×1 | acc[K] in VGPRs; A loaded once per k |
| L4 | `mxm_level4.h` / `transform_level4.h` | AMD MFMA (K=16,32) + L3 fallback | K²·8 B | 64×1×1 | `v_mfma_f64_16x16x4f64`; GFX90A/GFX940 only |
| L5 | `mxm_cublasdx.h` / `transform_cublasdx.h` | cuBLASDx Tensor Cores | device-managed | variable | NVIDIA only; double-buffered pipeline |
| L6 | `transform_kron.h` | Kronecker product GEMM | 0 | hipBLAS-managed | Single K³×K³ DGEMM; practical only for K≤16 |

**Default level**: L5 on CUDA, L3 on HIP (auto-selected if `-l` is not given).

### Level 3 — Register blocking (primary AMD path)

```
for i in 0..K²-1 (parallel over threads):
    acc[K] = 0          // register array, K doubles per thread
    for k in 0..K-1:
        aki = A[k, i]   // load A once per k
        for j in 0..K-1:
            acc[j] += aki * B[k, j]    // B already in LDS
    for j in 0..K-1:
        C[i, j] = acc[j]
```

K is a compile-time template parameter so each K value gets its own kernel binary. This is critical on AMD where register pressure is proportional to K.

### Level 4 — AMD MFMA

Uses `v_mfma_f64_16x16x4f64` (GFX90A / GFX940). One wavefront = 64 threads computes a 16×16 output tile with 4-deep contraction per instruction:

```
A fragment (16×4):  thread t → A[t/4][t%4]
B fragment (4×16):  thread t → B[t/16][t%16]
D output   (4 per thread): thread t → D[{(t/16)*4 + 0..3}][t%16]
```

Falls back to L3 for K not divisible by 16 (K=6,8,10,12,20).

### Level 6 — Kronecker GEMM

Collapses three sequential K²×K GEMMs into one K³×K³ DGEMM via:
```
vec(C) = (B^T ⊗ B^T ⊗ B^T) · vec(A)

KronMat[β,α] = B[α%K][β%K] · B[(α/K)%K][(β/K)%K] · B[α/K²][β/K²]
```
Built once on-device by `build_kron_kernel`; reused across all batches.  Memory: K⁶ × 8 bytes (128 MB at K=16, impractical beyond K≈16).

## Key Source Files

| File | Role |
|---|---|
| `transformbench.cu` | Main benchmark driver — option parsing, timing loop, FLOPs reporting |
| `transformbench.hip` | HIP wrapper (`#include transformbench.cu`) |
| `util.h` | Cross-platform macros: `CALL_KERNEL`, `MALLOC`, `MEMCPY_H2D`, `MEMCPY_D2H`, `CREATE_STREAM`, option parser |
| `validate_levels.hip` | Correctness test: any level vs L1 reference; `-l` selects level, `-K` and `-N` override defaults |

## Building

```bash
mkdir build && cd build

# AMD (MI250X / Frontier)
cmake .. \
  -DMRA_HAVE_HIP=1 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j

# NVIDIA (A100/H100, with cuBLASDx)
cmake .. \
  -DMRA_HAVE_CUDA=1 \
  -DMRA_HAVE_CUBLASDX=1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j
```

`Release` build type is required — `Debug` causes excessive register/memory use that breaks compilation.
CMake fetches cuBLASDx v25.06 automatically via FetchContent for the NVIDIA build.
If CMake cannot find HIP or hipBLAS config files, add `-DCMAKE_PREFIX_PATH=/opt/rocm-6.4.3`.

Symlink `compile_commands.json` to the project root for clangd IDE support:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

## Running

```bash
./transformbench_hip [options]
  -K <int>   transform order (default 16)
  -N <int>   number of tensors in batch (default 2048)
  -M <int>   max concurrent blocks (default 512)
  -n <int>   task submissions per timing rep (default 500)
  -r <int>   timing repetitions (default 5)
  -l <int>   optimization level 1-6 (default: auto)
  -s <int>   number of concurrent streams (default 4)

# Sweep levels for K=8
for L in 1 2 3 4 6; do ./transformbench_hip -K 8 -N 2048 -n 100 -l $L; done

# Sweep K at L3
for K in 6 8 10 12 16 20 32; do ./transformbench_hip -K $K -N 2048 -n 100 -l 3; done
```

Output (one line per timing rep):
```
Transform;level=L3-regblk;nfuncs=2048;nblocks=512;K=16;tasks=100;threads={128,1,1};smem=2048;Time(us)=12345;GFlop=403.0;Gflop/s=32.6
```

## FLOPs Accounting

- **L1–L4**: reported as `3 × 2 × K⁴ × nfuncs × ntasks` (mathematical minimum — useful throughput)
- **L6**: reported as `2 × K⁶ × nfuncs × ntasks` (actual GEMM work — inflated vs L1–L4 because K⁶ >> 3K⁴)

Do not compare L6 GFlop/s directly to L1–L4; it counts more FLOPs for the same mathematical result.

## Performance Notes

- **L1→L2**: ~K× speedup (eliminate B HBM redundancy)
- **L2→L3**: ~K× speedup (eliminate A HBM redundancy via register accumulation)
- **L3→L4**: ~2× on GFX90A (MFMA hardware matrix units, K=16 or K=32)
- **L6 vs L3** (K=6,8): 12–21× faster in wall-time despite more FLOPs — GPU utilization jumps from ~5% to ~80%

## Architecture Notes

- K-templated kernels (L3, L4): compile-time K avoids over-allocation of registers across K values
- One K³-sized workspace per block; workspace and output are ping-ponged across the three passes
- Multiple HIP streams (default 4) allow kernel overlap for throughput measurement
