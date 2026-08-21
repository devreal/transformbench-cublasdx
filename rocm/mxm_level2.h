#pragma once

#include "util.h"

/**
 * Level 2: B matrix loaded into LDS (shared memory) once per mTxmq call.
 *          A is streamed from global memory.  Threads are distributed over
 *          rows (i) rather than columns (j), so all 128 threads stay busy
 *          even for small K.
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K^2 x K  col-major   a[k,i] = a[k*dimi + i]
 *   B: K   x K  row-major   b[k,j] = b[k*dimj + j]
 *   C: K^2 x K  row-major   c[i,j] = c[i*dimj + j]
 */

namespace mra {

/* Public entry-point: always clears C (mTxmq semantics, equivalent to Q=true) */
template <typename aT, typename bT, typename cT>
__device__ void mTxmq_level2(size_type dimi, size_type dimj, size_type dimk,
                              cT* __restrict__ c, const aT* a, const bT* b) {
  extern __shared__ char smem_level2[];
  bT* b_shmem = reinterpret_cast<bT*>(smem_level2);

  /* Cooperatively load B (dimk * dimj elements) into LDS */
  for (int idx = threadIdx.x; idx < dimk * dimj; idx += blockDim.x) {
    b_shmem[idx] = b[idx];
  }
  __syncthreads();

  /* Each thread handles a stripe of rows; full j and k loops are sequential */
  for (size_type i = (size_type)threadIdx.x; i < dimi; i += (size_type)blockDim.x) {
    const aT* a_col_i = a + i;   /* pointer to a[0,i] in col-major layout */
    cT* ci = c + i * dimj;
    for (size_type j = 0; j < dimj; ++j) {
      cT sum = cT(0);             /* always clear: mTxmq semantics */
      const aT* aik = a_col_i;
      for (size_type k = 0; k < dimk; ++k, aik += dimi) {
        sum += (*aik) * b_shmem[k * dimj + j];
      }
      ci[j] = sum;
    }
  }
  __syncthreads();
}

template <typename T>
constexpr size_type mTxmq_level2_shmem_size(size_type K) {
  return K * K * sizeof(T);
}

template <typename T>
constexpr Dim3 mTxmq_level2_blockdim(int /*K*/) {
  return Dim3(MAX_THREADS_PER_BLOCK, 1, 1);
}

} // namespace mra
