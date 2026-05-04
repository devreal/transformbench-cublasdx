# Blocked 3D Transform — Corner-Turn Algorithm

A data-movement-conscious restructuring of the MRA 3D transform, designed so
each of K thread-block-equivalents ("wavefronts" on AMD) owns one K×K slab of
the tensor throughout the computation. Passes 1 and 2 are fully local per
block; pass 3 requires a single all-to-all "corner turn" that transposes the
block-level ownership from `k'` to `a`. Pass 3 right-multiplies by B so the
output lands in canonical (b, c) order with no post-transpose.

Reference implementations
- CPU (C++): `validate.hip` → `cpu_transform3d_blocked`
- NumPy: `mra_python/algorithms.py` → `transform_nd_blocked`

Both match `cpu_transform3d` / `transform_nd` bit-exactly (CPU) or to
floating-point noise (NumPy).

---

## The canonical operation

We compute the 3D transform

```
result[a, b, c] = Σ_{i, j, k} A[i, j, k] · B[i, a] · B[j, b] · B[k, c]
```

via three sequential contractions, one per tensor axis. Standard formulations
apply this as three K²×K GEMMs over the whole tensor, reloading the input
through HBM on every pass. The blocked algorithm keeps each K×K slab
resident in registers/LDS of a single wavefront for all three passes, paying
one inter-wave exchange instead of three HBM round-trips.

---

## Visual language

A consistent set of conventions used across every frame:

- **Axis colors.** `i'/a` red, `j'/b` green, `k'/c` blue.
- **Blocks are K×K squares.** Row-bar and column-bar outside each square
  name the axis currently playing that role.
- **Solid box = data, dashed box = operation.**
- **Arrows carry data.**

---

## Frame 1 — canonical operation

> ![Frame 1](frames/frame1_canonical.png)

A K×K×K input cube, a K×K matrix `B`, and a K×K×K output cube, with three
curved arrows labeled "contract `i'`", "contract `j'`", "contract `k'`".
Provides grounding; this is what all later frames compute.

---

## Frame 2 — distribute (slice along `k'`)

> ![Frame 2](frames/frame2_distribute.png)

The input cube is sliced along the fastest axis into K independent K×K
squares. Block `s` owns the slice `A[:, :, k'=s]`, with row = `i'` and
col = `j'`.

This establishes the row-of-K-squares motif used in every subsequent frame.

---

## Frame 3 — passes 1 and 2 (local GEMMs)

> ![Frame 3](frames/frame3_local_gemms.png)

Two stages stacked vertically on one frame:

```
row of K squares  ──(× B^T on left)──►  row of K squares       [Pass 1]
 rows: i',  cols: j'                     rows: a,   cols: j'

row of K squares  ──(× B on right)──►   row of K squares       [Pass 2]
 rows: a,   cols: j'                     rows: a,   cols: b
```

Both passes are **local** — no wavefront touches another wavefront's data.
A small "✓ local" badge on each block emphasizes this.

---

## Frame 4 — the corner turn (the money slide)

> ![Frame 4](frames/frame4_corner_turn.png)

The critical data-movement step. Shown as two rows of K blocks with colored
rows flowing between them.

**Before** (top): K blocks, each with rows colored red/green/blue/...
(= `a = 0, 1, 2, ...`). Every block has the same row-color pattern because
`a` is the within-block row index.

**After** (bottom): K blocks, each entirely one color. Block 0 all red,
block 1 all green, block 2 all blue. `a` is now the **block** index.
**Each arriving row is stored as a column**, so within each block the row
index is `b` and the column index is `k'`. This orientation sets pass 3
up to right-multiply by B and land in canonical order directly.

The operation is a **block-level transpose of a K×K super-matrix** whose
cells are K-vectors indexed by `b`, with an in-block transpose rolled into
the receive address (free — same data movement, different destination cell).

Callout on the frame: *"block-level transpose: block index ↔ within-block
row."* This is the single sentence the audience should leave with.

The main visual shows only the flow of one color (e.g. red rows, all
destined for block `a=0`) to keep the arrow count manageable; an inset
shows the full K=2 picture with all four arrows.

---

## Frame 5 — pass 3 (no un-shuffle needed)

> ![Frame 5](frames/frame5_pass3.png)

```
row of K squares  ──(× B on right)──►  canonical result
 rows: b,  cols: k'                     rows: b,   cols: c
```

Points to emphasize:

- Block index throughout this frame is `a` (inherited from the corner turn).
- **Right-multiply by B contracts `k'` → `c` in the column slot**, so the
  output is already in canonical (b, c) order. No per-block transpose on
  store — it was absorbed into the corner turn.
- Passes 2 and 3 now share the same GEMM orientation (B on the right);
  only pass 1 is the odd one out. One fewer MFMA microkernel variant to
  maintain on GPU.

---

## Frame 6 — the bookkeeping table (takeaway slide)

> ![Frame 6](frames/frame6_table.png)

| stage              | block idx | within-block row | within-block col |
|--------------------|-----------|------------------|------------------|
| distribute         | k'        | i'               | j'               |
| after pass 1       | k'        | **a**            | j'               |
| after pass 2       | k'        | a                | **b**            |
| after corner turn  | **a**     | **b**            | **k'**           |
| after pass 3       | a         | b                | **c**            |

Bold cells mark which slot changed on each row. Two annotations:

- Rows 1–2 (pass 1) and row 5 (pass 3): *"local GEMM: one slot updates."*
- Row 4: *"corner turn: block slot ↔ row slot, and row ↔ col inside the block."*

---

## Cost summary

| item                                 | cost per transform                    |
|--------------------------------------|---------------------------------------|
| compute (mathematical minimum)       | 3 · 2 · K⁴ FLOPs                      |
| HBM reads of A                       | K³ doubles (once, at distribute)      |
| HBM writes of result                 | K³ doubles (once, at pass-3 store)    |
| HBM reads of B                       | K² doubles (once, cached in LDS)      |
| inter-wave exchange (corner turn)    | K³ doubles (via LDS, one pass only)   |

Comparison against the naive 3-GEMM implementation is the whole point: HBM
traffic for A drops from 3·K³ to 1·K³, and the compute path is exactly the
same three K²×K contractions — just re-organized.

---

## K=2 walkthrough (self-checking)

Run `./validate -debug 1` in the build directory. Uses the counting tensor
`A[i, j, k] = 100i + 10j + k` and `B = I`, so every GEMM is a no-op and only
the corner turn produces visible motion. Each stage dumps all K blocks; the
final line is self-checking (`MATCH` / `DIFFER`).

Particularly useful as a "does this still work" smoke test when porting to
GPU.

---

## GPU implementation — measured perf on MI250X (K=16, one GCD)

Wired up as level 7. One block per tensor; grid = `nfuncs`.
Thread block = 64 × K threads = 1024 at K=16 (one wave per K×K slab).

Step-by-step results through the optimization arc (N=2048, FP64):

| version                                       | GF/s   | vs. prev |
|-----------------------------------------------|-------:|---------:|
| First cut (double-buffer LDS, B from HBM)     | 720    | —        |
| Single-buffer LDS + B cached in LDS           | 2270   | 3.15×    |
| + LDS row pad (kill 16-way bank conflict)     | 3320   | 1.46×    |
| + drop same-wave-only barriers (8 → 3)        | 3460   | 1.04×    |
| + fuse pass-2 store with corner-turn write    | 3475   | 1.00×    |
| + swap pass-1 operands → pass-2 reads acc     | 3475   | 1.00×    |
| + coalesced cooperative distribute            | 7400   | 2.13×    |
| + one block per tensor (grid = nfuncs)        | 8880   | 1.20×    |
| + `double4` loads for distribute + B cache    | **9340** | 1.05×  |

Final result: **9340 GF/s ≈ 20 % of scalar FP64 peak ≈ 10 % of MFMA peak**.
More importantly, **86 % of the HBM-bandwidth roofline** at the kernel's
arithmetic intensity (AI = 6.14 FLOPs/byte). See `roofline.py` / `roofline.png`.

### What we learned (not all optimizations paid off)

A few commits are worth flagging because their lesson matters more than
their number:

- **LDS bank-conflict pad — big win (1.46×)**. Classic 16-way conflict pattern
  in the corner turn. Simple row-stride pad from K to K+1.
- **LDS instruction cuts gave ~nothing**. Dropping barriers (-60 % LDS wait),
  fusing pass-2 into corner turn (-8 LDS ops/lane), register-fused pass-1→
  pass-2 (-8 more LDS ops/lane) all looked big in counters (40 % fewer LDS
  insts total) but moved wall clock by <5 %.  Interpretation: LDS was never
  on the critical path in this kernel; those ops were fully hidden in the
  shadow of MFMA/HBM latency.
- **Coalesced distribute — biggest single win (2.1× at N=2048)**. The prior
  per-wave stride-K reads were being served as individual cache lines by the
  hardware coalescer — 16× amplification on L2 request count.  `rocprof` showed
  HBM BW "only 26 % of peak" but actually the cache-line layer was saturated.
  Counter-intuitive: the HBM BW counter was misleading.
- **Widening the pass-3 store via LDS staging — regression (-17 %)**. The 4×
  narrow stores were already coalescing into 4 cache-line writes at the
  hardware level; adding a barrier + LDS round-trip lost more than the
  saved instructions gained.
- **MFMA swap trick (pass 1) — enables register fusion but no perf gain**.
  Swapping operands in pass 1 makes its output layout match pass 2's input
  exactly, so pass 2 reads acc directly.  Theoretically saves LDS traffic;
  wall-clock impact hidden by memory bubbles (see point 2 above).

### Compiler / ISA notes (GFX90A)

- MFMA `v_mfma_f64_16x16x4f64` layouts confirmed empirically via
  `test_mfma_layout.hip` (the upstream L4 comment had them wrong):
  - A_frag (16×4): lane t → A[t%16][t/16]
  - B_frag (4×16): lane t → B[t/16][t%16]
  - D output (16×16): lane t acc[e] → D[(t/16) + 4e][t%16]
- The widest global load/store on gfx90a is `global_{load,store}_dwordx4`
  (128-bit, 2 doubles per lane).  `double4` in code compiles to a pair
  of these.
- LDS bank period: 32 banks × 4 bytes = 128 bytes.  Stride-K doubles
  (K = 16) hits the same bank row → 16-way conflict; stride-(K+1) shifts
  by 2 banks → conflict-free.

---

## K=20 / K=24 — use v_mfma_f64_4x4x4f64 (design note)

K=20 and K=24 are the scientifically relevant odd-size cases (K=32 turns
out not to be; it was a suggestion just because the 16-divisibility
worked nicely with MFMA 16×16×4).  Neither K=20 nor K=24 divides by 16,
so they can't use `v_mfma_f64_16x16x4f64` directly without padding, and
padding to K=32 blows LDS.

The path forward is to use the smaller **`v_mfma_f64_4x4x4f64`** MFMA
variant — same hardware matrix core, 4×4 output tile instead of 16×16.
This avoids padding entirely because 4 divides both 20 and 24 cleanly.

### Tile-count cost

Per pass, per wave:

|       | 16×16×4 path (K=16) | 4×4×4 path (K=20) | 4×4×4 path (K=24) |
|-------|---------------------|--------------------|--------------------|
| output tiles per K slice | 1   | 25 = 5×5          | 36 = 6×6          |
| K slices needed          | 4   | 5                 | 6                 |
| MFMA calls per pass      | 4   | 125               | 216               |

The 4×4×4 approach uses ~30× more MFMA calls per pass than the K=16
kernel.  On gfx90a, matrix-core throughput for all f64 MFMA variants
is the same per-cycle-of-matrix-core, so more-but-smaller calls is a
wash on raw compute — but the **issue rate** may bottleneck at low
occupancy.

### 4×4×4 fragment layout — CONFIRMED via probe

`v_mfma_f64_4x4x4f64` on gfx90a returns **one double per lane**.
64 lanes × 1 = 64 output values = **four INDEPENDENT 4×4×4 GEMMs**
(NOT broadcast — earlier hypothesis disproved by `test_mfma_4x4x4_layout{,2,3}.hip`).

With the lane decomposition `S = 16·α + 4·g + β` where `β = S%4`,
`g = (S/4)%4` and `α = S/16`, the **group index** is `g` (∈ {0,1,2,3}).
Each of the four groups computes its own 4×4×4 GEMM `D_g = A_g · B_g`:

| fragment  | lane S holds                 |
|-----------|------------------------------|
| A (4×4)   | `A_g[m = β ][k = α]`         |
| B (4×4)   | `B_g[k = α ][n = β]`         |
| D (4×4)   | `D_g[m = α ][n = β]`         |

So A is column-major across 16 lanes, B row-major, D row-major, and
`t/16`/`(t/4)%4` each play different roles for the input vs. output
fragments — A's column index is where D's row index comes from, and
vice versa.  The four `g`-groups share nothing: `g=1` reads none of
`g=0`'s lanes.

This is strictly better for our K=20 case than 4×-broadcast would have
been: we can pack 4 independent 4×4 output tiles into a single
instruction, if we can lay out A/B so the four groups see different
tiles.

### Algorithm shape (once the 4×4×4 layout is understood)

Assuming we find a CBSZ/BLGP combination that gives 4 independent tiles:

- Replace the single 16×16 `mma_sync` in each of the three passes
  with a nested pair of tile loops:
  ```
  for (row_tile = 0; row_tile < K/4; ++row_tile)
    for (col_tile = 0; col_tile < K/4; ++col_tile)
      for (k_slice = 0; k_slice < K/4; ++k_slice)
        MFMA_4x4x4(...)
  ```
  At K=20: 5×5×5 = 125 calls per pass per wave.
- The per-wave output is K² = 400 (K=20) or 576 (K=24) doubles.
  400/64 = 6.25 per lane — **not integer**.  For K=20 this means
  some lanes will have 6 elements, others 7.  Awkward.  May want
  to pad to K=21 internally (6.56/lane)? Or group 2 MFMA calls
  per pair of lanes (?). Ugly.
  576/64 = 9 per lane — clean for K=24.
- LDS: K³ × 8 = 20³×8 = 64 KB (K=20), 24³×8 = 110 KB (K=24).
  K=20 just fits; K=24 still over.  So K=24 needs the same
  streaming story as K=32 (see section below).

### Recommended next steps (in order)

1. **Measure L3 at K=20 and K=24 first.** If L3 hits 50%+ of the HBM
   roofline at these K values, a custom MFMA kernel may not be worth
   the complexity.
2. **Pin down the 4×4×4 fragment layout** via a diagnostic kernel.
3. **Start with K=20 using 4×4×4** — clean LDS budget, even if awkward
   lane distribution.  K=24 can follow after streaming is figured out.
4. Consider a hybrid **16×16×4 main + 4×4×4 remainder** tiling (e.g.
   K=20 = 16+4 in each dim gives one 16×16 main tile + strips +
   corner).  Complex but avoids most of the 4×4×4 instruction overhead.

### K=20 status: implemented (MFMA via 4 independent tiles per call)

`transform_blocked_k20.h` ships both a scalar correctness baseline and
a 4×4×4 MFMA path.  Design choices and constraints:

| knob                    | K=20 choice                                           |
|-------------------------|-------------------------------------------------------|
| threads/block           | 64 × 10 = 640  (20 waves would exceed the 1024 cap)   |
| slabs per wave          | 2 (wave w owns k' ∈ {w, w+10})                        |
| LDS tensor buf          | K³ = 8000 doubles = 64 KB, **unpadded**               |
| B caching               | read from HBM (no room for B_lds at 64 KB cap)        |
| tile pattern            | 25 output tiles of 4×4 per slab, 7 MFMA rounds × 5 k-slices |
| wasted groups           | last round has 3 unused groups (fed zero)             |

Measured @ N=2048, FP64, MI250X single GCD:

| variant                               | GF/s | vs. L3  | kernel µs | MFMA busy | LDS wait/inst |
|---------------------------------------|-----:|--------:|----------:|----------:|--------------:|
| L3 (register-block)                   | 1967 | —       | —         | —         | —             |
| L7 scalar (K=20)                      | 1083 | 0.55×   | —         | —         | —             |
| L7 pure 4×4×4                         | 2787 | 1.42×   | 753       | 12.2 %    | 5.5 cyc       |
| L7 hybrid 16×16×4 + 4×4×4             | 3480 | 1.77×   | 591       | 11.1 %    | 19.4 cyc      |
| **L7 hybrid + fusion + wide dist**    | **6900** | **3.51×** | **370** | **17.8 %** | **4.1 cyc** |

The final configuration pairs three changes:

1. **Hybrid tiling** — 16×16×4 MFMA for the main 16×16 sub-tile, 4×4×4 MFMA
   for the 16×4 / 4×16 strips and the 4×4 corner.  43 % fewer MFMA issues
   than the pure 4×4×4 path.
2. **Pass 1 → Pass 2 register fusion** (operand-swap trick from the K=16
   kernel).  Pass 1 computes `D = temp1^T` so the main accumulator `p1_main[e]`
   at lane t directly equals `temp1[t%16][(t/16)+4e]` — exactly what pass 2's
   A-frag wants.  Pass 1 right strip gets the same treatment and feeds pass 2's
   5th k-slice from register `p1_right`.  Bottom strip and corner can't be
   fused cleanly (α/β transpose mismatch) and still round-trip through LDS.
   LDS instructions dropped 32 %, LDS wait cycles dropped 79 %.
3. **double4 distribute** — K³ = 8000 doubles = 2000 `double4` loads.

Remaining headroom (all less impactful than the above):
- **MFMA busy still 18 %** — there's still ~2× to gain before matching K=16's
  ~50 % MFMA-busy ceiling; further reductions in VALU overhead (mostly in
  the bottom/corner 4×4×4 loops) and in bank conflicts (currently 16-way on
  the stride-K=20 accesses for bottom/corner) would help.
- **No B_lds** — single-block-per-tensor has no room for a 3.2 KB B cache
  after the K³ = 64 KB `buf`.  Attempted 2-blocks-per-tensor with padded
  layout + B_lds but atomicAdd + pre-zero overhead (+500 MB HBM) erased the
  bank-conflict win — see notes below.
- **Pass 2 → Pass 3 fusion** isn't feasible: pass 3's cross-wave corner-turn
  REQUIRES data in LDS.

### Experiments that didn't pay off

- **2 blocks per tensor + K+1 padded LDS + atomicAdd** (`transform_kernel_blocked_k20_split`
  kept in the tree for reference).  Bank conflicts did drop to 0.2-way, but
  atomic read-modify-write on C and pre-zero memset inflated HBM traffic
  3× (258 → 752 MB), VALU went up 2.6×, and MFMA busy *fell* to 6 %.  Net:
  ~1.9× slower.  Take-away: bank conflict counters were real but not the
  critical path on the hybrid; the extra coordination costs of splitting
  the k'-axis across blocks easily dominated the LDS wins.

### 4×4×4 layout — CONFIRMED (overrides the "broadcast" hypothesis)

From `test_mfma_4x4x4_layout{,2,3}.hip`:

- The instruction is **4 INDEPENDENT 4×4×4 GEMMs**, not a 4-way
  broadcast.  Lane decomposition `S = 16α + 4g + β` with
  `β = S%4, g = (S/4)%4, α = S/16`.  `g` is the group id.
- A (4×4): lane S holds `A_g[m=β][k=α]`  (col-major)
- B (4×4): lane S holds `B_g[k=α][n=β]`  (row-major)
- D (4×4): lane S holds `D_g[m=α][n=β]`  (row-major)

Note α and β swap roles between input (A/B) and output (D) fragments —
same oddity as the 16×16×4 layout.

### Obvious next optimisations (not pursued here)

- **Fit B in LDS** via multi-block-per-tensor (e.g. 2 blocks handling
  10 k'-slabs each, atomic-add in pass 3).  LDS per block drops to
  ~35 KB and every MFMA can use LDS B.
- **Hybrid 16×16×4 main + 4×4×4 strip/corner** tile pattern.  Uses the
  faster instruction on the bulk of the work.
- **Pass-1/pass-2 register fusion** (as in the K=16 path).

---

## K=32 — design note (not yet implemented)

At K=32 the K=16 algorithm's working set doesn't fit in LDS, even split
across multiple blocks.  This is a real redesign, not a tweak.

### The budget problem

| thing                                          | K=16         | K=32         |
|------------------------------------------------|--------------|--------------|
| full K³ tensor                                 | 4 K dbl = 32 KB | 32 K dbl = **256 KB** |
| per-wave K×K slab                              | 256 dbl = 2 KB | 1024 dbl = 8 KB |
| 16 waves × slab (one block per tensor)         | 32 KB        | **128 KB**   |
| 16 waves × slab with K+1 pad                   | 34 KB        | **132 KB**   |
| LDS cap (MI250X)                               | 64 KB        | 64 KB        |

Even splitting into 2 blocks (each handling K/2 of k') gives 128 KB/block —
2× over cap.  There's no simple split that makes the K=16 algorithm fit.

### What must change

The working set per wave has to shrink from K×K to something smaller.
Two candidate approaches:

**(A) Each wave holds K × K/2 (half-slab).**
  - 2 waves cooperate on a full K×K slab.
  - Per-wave LDS: 512 dbl = 4 KB.  32 waves × 4 KB = 128 KB.  Still over.
  - 16 waves × 4 KB = 64 KB, halved coverage.  Would need 2 blocks to cover.

**(B) Stream the corner turn in k'-chunks.**
  - Hold only a chunk of the tensor at a time: K × K × (K/4) = 8 KB · 4 = 32 KB.
  - Pass 3 accumulates across 4 chunked corner turns into registers.
  - Extra HBM passes for re-reading A, unless we also chunk passes 1/2.
  - Complex but fits the budget.

**(C) Atomic-add reduction across 4 blocks.**
  - 4 blocks per tensor; each handles K/4 = 8 values of k'.
  - Each block's pass 3 contributes a partial; atomicAdd to HBM C.
  - 4 atomics per output element, 4K³ = 128K atomics per tensor.
  - Simpler code, slower arithmetic path.

### Thread-block sizing

At K=32, blockDim must be ≤ 1024 threads.  With 64-thread waves, that's
≤ 16 waves per block.  So we can't have 32 waves (one per k') in a single
block — each wave must handle ≥ 2 slabs, or blocks must handle fewer k'
values.

### Recommended approach for the next iteration

Start with **(C) — 4 blocks per tensor with atomic add**, because:
- Reuses the existing K=16 algorithm shape almost verbatim (per-block
  pipeline is identical).
- Simplest to get correct; good baseline before optimizing.
- Perf will be bad (atomics, 4× HBM writes for reduction) but it
  validates the design.

Then iterate toward **(B)** — streaming — which preserves single-block-
per-tensor semantics and should be faster once correct.

### Open questions worth probing first

- What does the actual MADNESS workload look like at K=32?  If `nfuncs`
  is small, block-count explosion from approach (C) isn't a problem
  but atomic contention might be.
- Does hipBLAS's batched DGEMM do better than anything we can write by
  hand at K=32?  Worth measuring L6 (kron) on real K=32 workloads as
  a reference point.
- L4 / L5 (upstream MFMA + rocWMMA paths) likely have K=32 code; are
  their designs worth stealing?

---

## References

- `validate.hip` — CPU reference and GPU L1 correctness check
- `validate_levels.hip` — multi-level correctness test (select L7 with `-l 7`)
- `transformbench.hip` — throughput benchmark (`-l 7`)
- `mra_python/algorithms.py::transform_nd_blocked` — NumPy version
- `test_mfma_layout.hip` — diagnostic that verifies MFMA fragment layouts
- `counters.txt`, `counters_deep.txt` — rocprof counter sets
- `roofline.py`, `roofline.png` — roofline analysis
- `transform.h`, `transform_level{2,3}.h` — earlier GPU levels for comparison
