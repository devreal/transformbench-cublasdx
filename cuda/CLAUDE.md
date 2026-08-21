# transformbench-cuda

CUDA-only port of the MRA (Multi-Resolution Analysis) transform benchmark — a
batched 3D tensor-times-matrix contraction. The parent directory holds the
dual-target (HIP + CUDA) original; this tree targets **NVIDIA only** and drops
every HIP branch. See [README.md](README.md) for the full porting rationale.

## Mathematical core

The transform applies a K×K matrix B (transposed) along each dimension of a
K×K×K tensor:

```
for d in {0, 1, 2}:
    C <- B^T x A    (in-place, cycling through a workspace)

GEMM shape per pass:
  A  (input):   K^2 x K  col-major
  B  (matrix):  K   x K  row-major
  C  (output):  K^2 x K  row-major
  FLOPs: 2 * K^2 * K * K per pass  ->  3 * 2 * K^4 per full transform
```

## Optimization levels

| Level | File | Technique | Threads | Notes |
|---|---|---|---|---|
| L1 | `mxm.h` / `transform.h` | Global memory only | K×min(K,128/K) | Correctness reference for `validate_levels` |
| L2 | `mxm_level2.h` / `transform_level2.h` | B cached in shared memory | 128 | Eliminates B HBM redundancy |
| L3 | `mxm_level3.h` / `transform_level3.h` | Register blocking (K-templated) | 128 | `acc[K]` in registers |
| L4 | `mxm_level4.h` / `transform_level4.h` | FP64 tensor cores, one warp | 32 | sm_80+; L3 fallback |
| L5 | `mxm_level5.h` / `transform_level5.h` | Tensor cores + A staged in smem | 256 | sm_80+; 8 warps |
| L6 | `mxm_wmma.h` / `transform_wmma.h` | `nvcuda::wmma`, warp per tile | 64–1024 | sm_80+; pads N to 8 |
| L7 | `mxm_level7.h` / `transform_level7.h` | B resident in registers, 3 GEMMs | 256 | sm_80+; K ∈ {8,16} |
| L8 | `transform_kron.h` | Kronecker product GEMM (cuBLAS) | cuBLAS | K⁶·8 B, caps near K=16 |
| L9 | `transform_cublasdx.h` | cuBLASDx, 3 GEMMs fused | cuBLASDx | K ∈ {8,10,16,20} |
| L10 | `transform_cublasdx_mxm.h` | cuBLASDx per-pass block GEMM | 128 | Uses `mra::mTxmq_cublasdx` |

**Default level**: L9 if cuBLASDx is available, else L3.

### FP64 tensor cores (levels 4, 5, 7)

NVIDIA's FP64 matrix instruction is `mma.sync.m8n8k4.f64` — an **8×8×4** tile
across a **32-lane warp**, against CDNA's 16×16×4 across a 64-lane wavefront.
All three levels go through `nvcuda::wmma` (see `dmma.h`) rather than raw PTX,
so `load_matrix_sync` owns the lane→element mapping. Constraints that shape the
code:

- shape must be 8×8×4; no other FP64 fragment geometry exists
- requires `__CUDA_ARCH__ >= 800`
- `ldm` for a `double` must be a multiple of 2 elements — **all padded shared
  memory strides must stay even** (L5 pads by +2, where the AMD source pads +1)
- every lane of the warp must reach `mma_sync`

`MRA_DMMA_SUPPORTED` (host, from `MRA_CUDA_ARCH`) and `MRA_HAVE_DMMA` (device,
from `__CUDA_ARCH__`) gate the paths; they must stay in agreement.

### Level 3 — register blocking (the portable workhorse)

```
for i in 0..K^2-1 (parallel over threads):
    acc[K] = 0          // register array, K doubles per thread
    for k in 0..K-1:
        aki = A[k, i]   // load A once per k
        for j in 0..K-1:
            acc[j] += aki * B[k, j]
    for j in 0..K-1:
        C[i, j] = acc[j]
```

K is a compile-time template parameter, so each K gets its own kernel binary and
register pressure stays proportional to K rather than max(K). On the
`transform-mem-alloc` branch the shared-memory staging of B in
`mTxmq_level3_k` is commented out — L3 reads B from global memory and relies on
L2 cache. `transform_level3_shmem_size` still reports K²·sizeof(T), so the
allocation is made but unused.

## Key source files

| File | Role |
|---|---|
| `transformbench.cu` | Benchmark driver — option parsing, timing loop, FLOPs reporting |
| `validate_levels.cu` | Correctness test: any level vs the L1 reference |
| `util.h` | Launch macros (`CALL_KERNEL`, `CONFIGURE_KERNEL`), memory macros, option parser |
| `dmma.h` | FP64 tensor-core support layer: fragment aliases, tile constants, availability macros |
| `mxm_cublasdx.h` | cuBLASDx `GEMMBuilder` + the `mTxmq_cublasdx` block GEMM |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j
```

`Release` is required — `Debug` inflates register/shared-memory use enough to
break compilation. CMake fetches cuBLASDx v25.06 via `FetchContent` unless
`-DUSE_CUBLASDX=OFF`. Use `-DCMAKE_CUDA_ARCHITECTURES=90` for Hopper.

Symlink `compile_commands.json` to this directory for clangd:

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

# Sweep levels for K=16
for L in 1 2 3 4 5 6 7 9 10; do ./transformbench_cuda -K 16 -N 2048 -n 100 -l $L; done

# Correctness
./validate_levels -l 5 -K 16
```

Output (one line per timing rep):

```
Transform;level=L3-regblk;nfuncs=2048;nblocks=512;K=16;tasks=100;threads={128,1,1};smem=2048;Time(us)=12345;GFlop=403.0;Gflop/s=32.6
```

## FLOPs accounting

- **L1–L7, L9, L10**: reported as `3 × 2 × K⁴ × nfuncs × ntasks` (mathematical
  minimum — useful throughput)
- **L8**: reported as `2 × K⁶ × nfuncs × ntasks` (actual GEMM work — inflated
  versus the rest because K⁶ ≫ 3K⁴)

Do not compare L8 GFlop/s directly to the other levels.

## Architecture notes

- K-templated kernels (L3–L7): compile-time K avoids over-allocating registers
  across K values
- One K³-sized workspace per block; workspace and output are ping-ponged across
  the three passes — except L7, which keeps all three passes inside one kernel
  call so B never leaves registers
- L7 reuses a single K³ shared buffer in place via the MADNESS pointer trick:
  C written row-major as `buf[i*K+j]` is reread col-major as `buf[k*K²+i]`.
  Never pad that buffer — the two views must alias exactly.
- Multiple streams (default 4) allow kernel overlap for throughput measurement
