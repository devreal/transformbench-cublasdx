# transformbench — CUDA port

A CUDA-only rewrite of the MRA transform benchmark that lives in the parent
directory. Same mathematical kernel, same measurement harness, same output
format; every HIP/ROCm construct has been replaced with its CUDA equivalent,
and the AMD matrix-core levels have been re-derived for NVIDIA FP64 tensor
cores rather than mechanically translated.

## The kernel being measured

A batched 3D tensor-times-matrix contraction: a K×K matrix B is applied
(transposed) along each dimension of a K×K×K tensor.

```
for d in {0, 1, 2}:
    C <- B^T x A    (cycling through a workspace)

GEMM shape per pass:
  A (input):   K^2 x K  col-major
  B (matrix):  K   x K  row-major
  C (output):  K^2 x K  row-major
  FLOPs: 2 * K^2 * K * K per pass  ->  3 * 2 * K^4 per full transform
```

## Optimization levels

| Level | Files | Technique | Threads | Needs |
|---|---|---|---|---|
| L1 | `mxm.h` / `transform.h` | Global memory only — correctness reference | K×min(K,128/K) | — |
| L2 | `mxm_level2.h` / `transform_level2.h` | B cached in shared memory | 128 | — |
| L3 | `mxm_level3.h` / `transform_level3.h` | Register blocking, `acc[K]` in registers | 128 | — |
| L4 | `mxm_level4.h` / `transform_level4.h` | FP64 tensor cores, one warp | 32 | sm_80 |
| L5 | `mxm_level5.h` / `transform_level5.h` | FP64 tensor cores, A staged in shared memory | 256 | sm_80 |
| L6 | `mxm_wmma.h` / `transform_wmma.h` | `nvcuda::wmma`, one warp per output tile | 64–1024 | sm_80 |
| L7 | `mxm_level7.h` / `transform_level7.h` | FP64 tensor cores, B resident in registers | 256 | sm_80 |
| L8 | `transform_kron.h` | Single K³×K³ DGEMM via Kronecker product | cuBLAS | — |
| L9 | `transform_cublasdx.h` | cuBLASDx, three GEMMs fused in shared memory | cuBLASDx-chosen | cuBLASDx |
| L10 | `transform_cublasdx_mxm.h` | cuBLASDx as a per-pass block GEMM | 128 | cuBLASDx |

Default level: **L9** when cuBLASDx is available, otherwise **L3**.

Levels 4–7 fall back to the L3 register-blocking kernel whenever the target is
older than sm_80 or K does not admit a whole tiling; the fallback is transparent
and the reported level name does not change.

## How the AMD levels were ported

The bulk of the tree is a direct translation — `hipMalloc`→`cudaMalloc`,
`hipStream_t`→`cudaStream_t`, `__HIP_DEVICE_COMPILE__`→`__CUDA_ARCH__`,
hipBLAS→cuBLAS. Levels 4–7 are not, because they are built on AMD matrix cores
whose geometry has no NVIDIA twin:

| | CDNA (gfx90a/gfx940) | NVIDIA (sm_80+) |
|---|---|---|
| instruction | `v_mfma_f64_16x16x4f64` | `mma.sync.m8n8k4.f64` |
| tile (M×N×K) | 16 × 16 × 4 | 8 × 8 × 4 |
| lanes cooperating | 64 (wavefront) | 32 (warp) |
| accumulators per lane | 4 | 2 |

The AMD sources index MFMA operands by hand — thread *t* supplies `A[t/4][t%4]`,
holds output rows `(t/16)*4 + 0..3`, and so on. That lane mapping is specific to
CDNA. The ports drive the tensor cores through `nvcuda::wmma` instead, so
`load_matrix_sync` owns the lane→element mapping; this keeps the translation
correct without hard-coding a register layout. What each level *demonstrates* is
preserved exactly — who stages A, where B lives, how many warps cooperate. See
[dmma.h](dmma.h) for the shared support layer.

Consequences worth knowing:

- **Tile size halves (16→8)**, so every tile-count constant is re-derived. L4/L5/L7
  need K to be a multiple of 8 (K = 8, 16, 32) where the AMD versions need 16.
- **L6 column padding drops from 16 to 8.** An 8-wide fragment expresses a K < 16
  GEMM directly, so rocWMMA's LDS-resident small-K special case (`transform_klt16`)
  has no counterpart and is not carried over. L6 also stages output **per warp**
  (8×8 scratch) instead of staging the whole M×N_PAD output as rocWMMA does —
  the full-output variant would need ~75 KB of shared memory at K=20.
- **L5's shared-memory padding is +2, not +1.** The WMMA API requires a `double`
  leading dimension to be a multiple of 16 bytes / 8 bytes = 2 elements, so the
  AMD +1 pad would be rejected. Chunking is otherwise identical: K=16 → 4 strips
  of 64 rows, K=32 → 8 strips of 128.
- **L7 drops the XOR swizzle.** `load_matrix_sync`/`store_matrix_sync` compute
  their own lane addresses from a base pointer and a stride, leaving nowhere to
  inject an address permutation, and padding would break the pointer trick that
  the in-place buffer reuse depends on. The K²-stride bank conflicts the AMD
  swizzle avoids are therefore accepted here. Correctness is unaffected;
  L7 shared-memory throughput is the thing to watch when profiling.
- **L7 is limited to K ∈ {8, 16}.** A warp holds its whole A partition in
  registers; at K=32 that is 128 fragments per lane and would spill. Other K
  values are dispatched to L3 by the host-side submit function.

## Deliberate differences from the parent tree

These are fixes, not translation artifacts — each one is a place where the
dual-target sources had drifted.

1. **The level map is complete.** `transformbench.cu` upstream includes
   `transform_cublasdx.h` but never dispatches to it, and its `level_names[]`
   still describes level 5 as cuBLASDx while case 5 calls the AMD MFMA kernel.
   Here levels 1–8 keep their upstream meaning and cuBLASDx gets levels 9 and 10.
2. **L1 is unambiguously the reference.** Upstream, `transform.h` includes
   `mxm_cublasdx.h`, which declares `mTxmq(long, long, long, …)`; every call site
   passes `int`, which binds exactly to the `size_type` overload in `mxm.h`, so
   the cuBLASDx path was silently unreachable. `mxm_cublasdx.h`'s entry point is
   renamed `mTxmq_cublasdx` here and reached through level 10 on purpose, leaving
   L1 a genuine global-memory reference for `validate_levels`.
3. **L7's fallback runs all three passes.** The AMD `mTxmq_level7_k` fallback
   calls `mTxmq_level3_impl` once, but its caller expects the full three-GEMM
   chain, so a non-MFMA target silently produced a one-pass result. The CUDA
   fallback ping-pongs through the workspace like L3 does.
4. **`gemm7_pass` no longer marks aliasing pointers `__restrict__`.** GEMM 2
   passes the same buffer as source and destination.
5. **Streams are destroyed** at the end of `transform_bench`.

`mxm_level3.h` is carried over verbatim, including the commented-out staging
copy on the `transform-mem-alloc` branch — L3 currently reads B straight from
global memory and relies on L2 cache. The comment in `transform_level3.h` was
corrected to say so.

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j
```

`Release` is the default and is required — `Debug` inflates register and
shared-memory use enough to break compilation.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `80` | `90` for Hopper. Below 80 disables FP64 tensor cores; L4–L7 then run their L3 fallback. |
| `USE_CUBLASDX` | `ON` | Fetches cuBLASDx v25.06 via `FetchContent`. `OFF` drops levels 9 and 10. |
| `USE_SUGGEST_LAYOUT` | `ON` | cuBLASDx suggested layout instead of `get_layout`. |
| `DEBUG_TENSOR_TYPE` | `OFF` | Compile-time print of cute tensor types (breaks the build by design). |

For clangd:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

## Running

```bash
./transformbench_cuda [options]
  -K <int>   transform order (default 16)
  -N <int>   number of tensors in batch (default 2048)
  -M <int>   max concurrent blocks (default 512)
  -n <int>   task submissions per timing rep (default 500)
  -r <int>   timing repetitions (default 5)
  -l <int>   optimization level 1-10 (default: auto)
  -s <int>   number of concurrent streams (default 4)

# Sweep levels at K=16
for L in 1 2 3 4 5 6 7 9 10; do ./transformbench_cuda -K 16 -N 2048 -n 100 -l $L; done

# Sweep K at L3
for K in 6 8 10 12 16 20 32; do ./transformbench_cuda -K $K -N 2048 -n 100 -l 3; done
```

Output, one line per timing rep:

```
Transform;level=L3-regblk;nfuncs=2048;nblocks=512;K=16;tasks=100;threads={128,1,1};smem=2048;Time(us)=12345;GFlop=403.0;Gflop/s=32.6
```

## Correctness

```bash
./validate_levels [-l <level>] [-K <k>] [-N <nfuncs>]
```

Compares any level against the L1 reference; with no `-K` it sweeps
K ∈ {6, 8, 10, 12, 16}. Passing means max relative error < 1e-10.
Levels with narrower dispatch tables (L7: K ∈ {8,16}; L9: K ∈ {8,10,16,20})
report the K values they do not handle.

## K support by level

| Level | K values on the accelerated path | Other K |
|---|---|---|
| L1–L3 | 6, 8, 10, 12, 16, 20, 32 | — |
| L4, L5 | 8, 16, 32 | L3 fallback |
| L6 | 4, 8, 12, 16, 20, 32 (any K % 4 == 0) | L3 fallback |
| L7 | 8, 16 | dispatched to L3 |
| L8 | any, but K⁶·8 bytes caps it near K=16 | — |
| L9 | 8, 10, 16, 20 | throws |
| L10 | 6, 8, 10, 12, 16, 20, 32 | prints diagnostic |

## FLOPs accounting

- **L1–L7, L9, L10**: reported as `3 × 2 × K⁴ × nfuncs × ntasks` — the
  mathematical minimum, i.e. useful throughput.
- **L8**: reported as `2 × K⁶ × nfuncs × ntasks` — the actual GEMM work, which is
  inflated relative to the others because K⁶ ≫ 3K⁴.

Do not compare L8 GFlop/s directly against the other levels; it counts far more
FLOPs for the same mathematical result.

## Build verification status

The port has been checked by parsing every translation unit and force-instantiating
every device entry point for all supported K, with both the tensor-core paths and
the fallback paths enabled. It has **not** been compiled with `nvcc` or run on a
GPU — no CUDA toolkit was available in the environment where it was written. The
`nvcuda::wmma` fragment/layout combination used here (`matrix_a` col_major,
`matrix_b` row_major, FP64 8×8×4) is the one the rocWMMA source uses and is
within the documented API, but it is the first thing to check if a build fails.
