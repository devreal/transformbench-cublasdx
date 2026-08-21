#ifndef MRA_MXM_H
#define MRA_MXM_H

#include "util.h"


#if __has_include(<blas.hh>)
#include <blas.hh>
#define HAVE_BLASPP 1
#endif // __has_include(<blas.hh>)

namespace mra {


#ifndef MRA_HAVE_MTXM

#if defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)
  /**
   * blaspp implementation of A^T * B
   * c(i,j) += sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  void mTxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans,
               dimi, dimj, dimk,
               1.0, a, dimi, b, dimj,
               Q ? 0.0 : 1.0, c, dimj);
  }
#else  // HAVE_BLASPP
  /**
   * reference implementation, adapted from madness
   * c(i,j) += sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *aik_ptr = a + i;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }

        for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
          aT aki = *aik_ptr;
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] += aki*b[k*dimj+j];
          }
        }
      }
    }
    SYNCTHREADS();
  }
#endif // HAVE_BLASPP

  template<typename T>
  constexpr size_type mTxm_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mTxm_blockdim(int K) {
    return max_thread_dims(K);
  }


#endif // MRA_HAVE_MTXM

#ifndef MRA_HAVE_MTXMQ

  /**
   * blaspp implementation of A^T * B
   * c(i,j) = sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mTxmq(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    mTxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b);
  }

  template<typename T>
  constexpr size_type mTxmq_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mTxmq_blockdim(int K) {
    return mTxm_blockdim<T>(K);
  }

#endif // MRA_HAVE_MTXMQ


#ifndef MRA_HAVE_MXM

#if defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  /**
   * blaspp implementation of A * B
   * c(i,j) += sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  void mxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               dimi, dimj, dimk,
               1.0, a, dimk, b, dimj,
               Q ? 0.0 : 1.0, c, dimj);
  }
#else // defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)
  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxm(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b) {
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          for (long k=0; k<dimk; ++k) { /* not parallelized */
            ci[j] += ai_ptr[k]*b[k*dimj+j];
          }
        }
      }
    }
    SYNCTHREADS();
  }
#endif // defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  template<typename T>
  constexpr size_type mxm_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mxm_blockdim(int K) {
    return max_thread_dims(K);
  }

#endif // MRA_HAVE_MXM


#ifndef MRA_HAVE_MXMQ

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmq(size_type dimi, size_type dimj, size_type dimk,
                  cT* __restrict__ c, const aT* a, const bT* b) {
    mxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b);
  }

  template<typename T>
  constexpr size_type mxmq_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mxmq_blockdim(int K) {
    return mxm_blockdim<T>(K);
  }

#endif // MRA_HAVE_MXMQ


#ifndef MRA_HAVE_MXMT

#if defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  /**
   * blaspp implementation of A * B^T
   * c(i,j) += sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  void mxmT(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::Trans,
               dimi, dimj, dimk,
               1.0, a, dimk, b, dimk,
               Q ? 0.0 : 1.0, c, dimj);
  }

#else // defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmT(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b) {
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          cT sum = 0.0;
          for (size_type k=0; k<dimk; ++k) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose b first into shared memory
             */
            sum += ai_ptr[k]*b[j*dimk+k];
          }
          if constexpr (Q) {
            ci[j] = sum;
          } else {
            ci[j] += sum;
          }

        }
      }
    }
    SYNCTHREADS();
  }

#endif // defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  template<typename T>
  constexpr size_type mxmT_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mxmT_blockdim(int K) {
    return max_thread_dims(K);
  }

#endif // MRA_HAVE_MXMT


#ifndef MRA_HAVE_MXMTQ

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mxmTq(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b) {
    mxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b);
  }

  template<typename T>
  constexpr size_type mxmTq_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mxmTq_blockdim(int K) {
    return mxmT_blockdim<T>(K);
  }


#endif // MRA_HAVE_MXMTQ



#ifndef MRA_HAVE_MTXMT

#if defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  /**
   * blaspp implementation of A^T * B^T
   * c(i,j) += sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  void mTxmT(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b) {
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::Trans,
               dimi, dimj, dimk,
               1.0, a, dimi, b, dimj,
               Q ? 0.0 : 1.0, c, dimj);
  }

#else

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxmT(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b) {
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          const aT *aik_ptr = a + i;
          for (long k=0; k<dimk; ++k,aik_ptr+=dimj) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose a and b first into shared memory
             */
            ci[j] += *aik_ptr * b[j*dimk+k];
          }
        }
      }
    }
    SYNCTHREADS();
  }

#endif // defined(HAVE_BLASPP) && !defined(HAVE_DEVICE_ARCH)

  template<typename T>
  constexpr size_type mTxmT_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mTxmT_blockdim(int K) {
    return max_thread_dims(K);
  }


#endif // MRA_HAVE_MTXMT



#ifndef MRA_HAVE_MTXMTQ

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mTxmTq(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b) {
    mTxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b);
  }

  template<typename T>
  constexpr size_type mTxmTq_shmem_size(size_type K) {
    return 0;
  }

  template<typename T>
  constexpr Dim3 mTxmTq_blockdim(int K) {
    return mTxmT_blockdim<T>(K);
  }

#endif // MRA_HAVE_MTXMTQ

} // namespace mra

#endif // MRA_MXM_H
