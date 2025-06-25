#ifndef MRA_OPS_MXM_CUBLASDX_H
#define MRA_OPS_MXM_CUBLASDX_H

#include "util.h"

/**
 * An implementation of A^T x B using cublasdx.
 * We assume that A is tall-and-skinny (K^2 x K) and B is square (K x K).
 * There is some code to cover the case where A is square and B is wide-and-skinny
 * but that is not yet implemented and we don't use it yet.
 */

#define MRA_CUBLASDX_BLOCK_C 0

#if 1 || __has_include(<cublasdx.hpp>)

#define MRA_HAVE_CUBLASDX 1

#if !defined(MRA_CUDA_ARCH) || MRA_CUDA_ARCH < 70
#error "MRA_CUDA_ARCH must be defined and >= 70 to use cublasdx"
#endif

#include <cublasdx.hpp>

#if MRA_CUDA_ARCH == 70
#define MRA_CUBLASDX_SM 700
#define MRA_CUBLASDX_MAX_SHM (30*1024)
#elif MRA_CUDA_ARCH == 80
#define MRA_CUBLASDX_SM 800
#define MRA_CUBLASDX_MAX_SHM (40*1024)
#elif MRA_CUDA_ARCH == 90
#define MRA_CUBLASDX_SM 900
#define MRA_CUBLASDX_MAX_SHM (110*1024)
#else
#warning "Unknown MRA_CUDA_ARCH for cublasdx, using 80"
#define MRA_CUBLASDX_SM 800
#endif

#ifdef DEBUG_TENSOR_TYPE
#define PRINT_TENSOR_TYPE(t) cute::print_type(t)
#else  // DEBUG_TENSOR_TYPE
#define PRINT_TENSOR_TYPE(t)
#endif // DEBUG_TENSOR_TYPE


namespace mra {

  namespace detail {

    constexpr int CUBLAS_MIN_MN = 16;

    template<typename T, int K>
    constexpr int cublasdx_max_mn() {
      // K^2 for square B/A, double buffering for A/B and C
      auto max_nm = ((MRA_CUBLASDX_MAX_SHM / sizeof(T)) - K*K) / ((3+MRA_CUBLASDX_BLOCK_C)*K);
      // round down to the nearest power of 2
      // TODO: std::log2 is constexpr only since C++26
      //int p = std::pow(2, (int)std::log2(max_nm));
      int l = 1;
      while ((l<<1) <= max_nm) l <<= 1;
      return std::min(l, K*K);
    }

    template<typename T, int N, int M, int K,
             cublasdx::arrangement ArrangeA,
             cublasdx::arrangement ArrangeB,
             cublasdx::arrangement ArrangeC>
    struct GEMMBuilder {

    private:
      using BaseGEMM = decltype(cublasdx::Precision<T, T, T>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<MRA_CUBLASDX_SM>() // TODO
                              + cublasdx::Block()
                              + cublasdx::MaxAlignment());
      using GEMM_ = decltype(BaseGEMM() + cublasdx::Size<N, M, K>()
                            + cublasdx::Arrangement<ArrangeA, ArrangeB, ArrangeC>());
      using GEMM_suggested_ld = cublasdx::suggested_leading_dimension_of_t<GEMM_, MRA_CUBLASDX_SM>;
    public:
      using GEMM = decltype(GEMM_() + GEMM_suggested_ld());
    };

    template<typename GEMM>
    __device__ void mTxmq_cublasdx_core(auto&& a_shared_tensor, auto&& b_shared_tensor,
                                        auto&& c_tensor,
                                        auto&& load = [](){}, auto&& prefetch = [](){}) {

      using alignment = cublasdx::alignment_of<GEMM>;

      /* load data to shared memory */
      load();
      /* wait for load to complete */
      cublasdx::copy_wait();

      /* prefetch data for next iteration */
      prefetch();

      // Execute using register API
      auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

      // Store back to global memory using cublasdx::copy_fragment API

      cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_tensor, partitioner);
    }

    /**
     * Compute the shared memory requirements for a given GEMM.
     * Takes into account double buffering of A (block_a) and B (block_b) as well as
     * staging of results through shared memory (block_c).
     */
    template<typename GEMM>
    constexpr int cublasdx_shmem_size_for(bool block_a, bool block_b, bool block_c) {
      auto calc = cublasdx::make_shared_storage_calculator()
                  .add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::get_layout_smem_a())
                  .add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::get_layout_smem_b());
      if (block_a) {
        calc.add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::get_layout_smem_a());
      }
      if (block_b) {
        calc.add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::get_layout_smem_b());
      }
      if (block_c) {
        // double buffering of C
        calc.add(cublasdx::alignment_of_v_c<GEMM>, sizeof(typename GEMM::c_value_type), GEMM::get_layout_smem_c());
        calc.add(cublasdx::alignment_of_v_c<GEMM>, sizeof(typename GEMM::c_value_type), GEMM::get_layout_smem_c());
      }

      int shared_memory_size = calc.get();
      return shared_memory_size;
    }

    template<typename T, int K>
    constexpr int cublasdx_shmem_size_k() {
      constexpr auto blockdims = max_thread_dims(K);
      using BaseGEMM = decltype(cublasdx::Precision<T>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<MRA_CUBLASDX_SM>() // TODO
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment());
      constexpr auto max_mn = cublasdx_max_mn<T, K>();
      using GEMMBlockA = typename GEMMBuilder<T, std::min(max_mn, K*K), K, K, cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>::GEMM;
      auto size = cublasdx_shmem_size_for<GEMMBlockA>(true, false, MRA_CUBLASDX_BLOCK_C);
      return size;
    }

    template<int M, int N, int K, typename T>
    __device__ void mTxmq_cublasdx_block(T* c, const T* a, const T* b) {
      constexpr auto blockdims = max_thread_dims(K);
      extern __shared__ __align__(16) char smem[];
      constexpr auto max_mn = cublasdx_max_mn<T, K>();
      /* assuming aT = bT = cT for now */
      using GEMM = typename GEMMBuilder<T, std::min(max_mn, M), std::min(max_mn, N), K, cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>::GEMM;

      using alignment = cublasdx::alignment_of<GEMM>;


      if constexpr (M == K*K) {
        constexpr auto num_iter = M/max_mn;
        //if (is_team_lead()) printf("mTxmq_cublasdx_block: max_mn %d, shared_memory %u, smem %p, M = %d, N = %d, K = %d iter %d\n", max_mn, cublasdx_shmem_size_for<GEMM>(true, false, true), smem, M, N, K, num_iter);
        //__syncthreads();

        if constexpr (num_iter > 0) {
          auto [smem_a, smem_b, smem_a_n, smem_c, smem_c_n] =
            cublasdx::shared_memory::slice_into_pointers<GEMM::a_value_type, GEMM::b_value_type, GEMM::a_value_type,
                                                         GEMM::c_value_type, GEMM::c_value_type>(
                smem,
                cublasdx::alignment_of_v_a<GEMM>, cublasdx::cosize(GEMM::get_layout_smem_a()),
                cublasdx::alignment_of_v_b<GEMM>, cublasdx::cosize(GEMM::get_layout_smem_b()),
                cublasdx::alignment_of_v_a<GEMM>, cublasdx::cosize(GEMM::get_layout_smem_a()),
                cublasdx::alignment_of_v_c<GEMM>, cublasdx::cosize(GEMM::get_layout_smem_c()),
                cublasdx::alignment_of_v_c<GEMM>, cublasdx::cosize(GEMM::get_layout_smem_c()));

          /* copy b tensor into shared memory and leave there */
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
          cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
          PRINT_TENSOR_TYPE(b_global_tensor);
          PRINT_TENSOR_TYPE(b_shared_tensor);

          auto a_shared_tensor   = cublasdx::make_tensor(smem_a,   GEMM::get_layout_smem_a());
          auto a_shared_tensor_n = cublasdx::make_tensor(smem_a_n, GEMM::get_layout_smem_a());

          auto c_shared_tensor   = cublasdx::make_tensor(smem_c,   GEMM::get_layout_smem_c());
          auto c_shared_tensor_n = cublasdx::make_tensor(smem_c_n, GEMM::get_layout_smem_c());

          int i; // used past the for loop below

          auto make_c_global_tensor = [&](int i){
            return cublasdx::make_tensor(c+((i*max_mn)*N), GEMM::get_layout_gmem_c());
          }

          auto store_c = [&](auto& c_shared_tensor) {
#if MRA_CUBLASDX_BLOCK_C
            __syncthreads(); // make sure prior computations are done
            auto c_global_tensor = make_c_global_tensor(i-1);
            cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
#endif  // MRA_CUBLASDX_BLOCK_C
          };
          for (i = 0; i < num_iter; i++) {
            // Make global memory tensors
            auto a_global_tensor = cublasdx::make_tensor(a+(i*max_mn),     GEMM::get_layout_gmem_a(cute::Int<M>{}));
            PRINT_TENSOR_TYPE(a_global_tensor);
            PRINT_TENSOR_TYPE(a_shared_tensor);
            PRINT_TENSOR_TYPE(make_c_global_tensor(i));
            PRINT_TENSOR_TYPE(c_shared_tensor);
            //auto c_global_tensor = cublasdx::make_tensor(c+((i*max_mn)*N), GEMM::get_layout_gmem_c());
            mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor,
#if MRA_CUBLASDX_BLOCK_C
                                      c_shared_tensor,
#else  // MRA_CUBLASDX_BLOCK_C
                                      /* global tensor */
                                      make_c_global_tensor(i)
#endif // MRA_CUBLASDX_BLOCK_C
                                      [&](){
                                        /* load only on first iteration, all others are prefetched */
                                        if (i == 0) {
                                          //if (is_team_lead()) printf("Loading initial block %d\n", i);
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                        }
                                      },
                                      [&](){
                                        /* store prior iteration's result */
                                        if (i > 0) {
                                          //if (is_team_lead()) printf("Storing block %d\n", i-1);
                                          store_c(c_shared_tensor_n);
                                        }
                                        /* prefetch into shared memory */
                                        if ((i+1) < num_iter) {
                                          //if (is_team_lead()) printf("Prefetching block %d\n", i);
                                          auto a_global_tensor = cublasdx::make_tensor(a+((i+1)*max_mn), GEMM::get_layout_gmem_a(cute::Int<M>{}));
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor_n);
                                        }
                                      });
            std::swap(a_shared_tensor, a_shared_tensor_n);
            std::swap(c_shared_tensor, c_shared_tensor_n);
          }
          /* store the last block of C */
          store_c(c_shared_tensor_n);
        }

        /* handle remainder */
        constexpr const auto R = M%max_mn;
        if constexpr (0 < R) {
          // Make global memory tensors
          using GEMM = typename GEMMBuilder<T, R, N, K, cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>::GEMM;
          auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem, GEMM::get_layout_smem_a(),
                                                                                    GEMM::get_layout_smem_b(),
                                                                                    GEMM::get_layout_smem_c());
          auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
          auto a_global_tensor = cublasdx::make_tensor(a+((M/max_mn)*max_mn),   GEMM::get_layout_gmem_a(cute::Int<M>{}));
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
          auto c_global_tensor = cublasdx::make_tensor(c+((M/max_mn)*max_mn*N), GEMM::get_layout_gmem_c());
          auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());
          mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor,
#if MRA_CUBLASDX_BLOCK_C
                                      c_shared_tensor,
#else  // MRA_CUBLASDX_BLOCK_C
                                      c_global_tensor,
#endif // MRA_CUBLASDX_BLOCK_C
                                    [&](){
                                      cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                      cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
                                    },
                                    [](){});
          /* move the C block back to global memory */
          cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
        }
      } else {
        // TODO: implement!
        static_assert(M == K*K, "N equal to K*K currently not supported");
      }
      /* final sync */
      cublasdx::copy_wait();
    }

  } // namespace detail

  template <typename aT, typename bT, typename cT>
  __device__ void mTxmq(long dimi, long dimj, long dimk,
                        cT* c, const aT* a, const bT* b) {
    int M = dimi;
    int N = dimj;
    int K = dimk;
    if (M == K*K) {
      // A is tall and skinny, B is square
      if (K == 6) {
        detail::mTxmq_cublasdx_block<36, 6, 6>(c, a, b);
      } else if (K == 8) {
        detail::mTxmq_cublasdx_block<64, 8, 8>(c, a, b);
      } else if (K == 10) {
        detail::mTxmq_cublasdx_block<100, 10, 10>(c, a, b);
      } else if (K == 12) {
        detail::mTxmq_cublasdx_block<12*12, 12, 12>(c, a, b);
      } else if (K == 16) {
        detail::mTxmq_cublasdx_block<16*16, 16, 16>(c, a, b);
      } else if (K == 20) {
        detail::mTxmq_cublasdx_block<400, 20, 20>(c, a, b);
      } else if (K == 32) {
        detail::mTxmq_cublasdx_block<32*32, 32, 32>(c, a, b);
      } else {
        if (is_team_lead()) printf("mTxmq: Unsupport K = %d\n", K);
      }
    } else {
        printf("mTxmq: Unknown configuration with M = %d, N = %d, K = %d\n", M, N, K);
    }
    /* make sure all is done */
    __syncthreads();
  }

  template<typename T>
  constexpr int mTxmq_shmem_size(int K) {
    switch (K) {
      case 6: return detail::cublasdx_shmem_size_k<T, 6>();
      case 8: return detail::cublasdx_shmem_size_k<T, 8>();
      case 10: return detail::cublasdx_shmem_size_k<T, 10>();
      case 12: return detail::cublasdx_shmem_size_k<T, 12>();
      case 16: return detail::cublasdx_shmem_size_k<T, 16>();
      case 20: return detail::cublasdx_shmem_size_k<T, 20>();
      case 32: return detail::cublasdx_shmem_size_k<T, 32>();
      default: THROW("CUBLASdx: Unsupported K");
    }
  }


  namespace detail {
    template<typename T, int K>
    constexpr Dim3 cublasdx_blockdim_k() {

      return Dim3(MAX_THREADS_PER_BLOCK, 1, 1);
      constexpr auto max_mn = cublasdx_max_mn<T, K>();
      using GEMM = typename GEMMBuilder<T, std::min(max_mn, K*K), K, K, cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>::GEMM;
      return GEMM::suggested_block_dim;
    }

  } // namespace detail
  template<typename T>
  constexpr Dim3 mTxmq_blockdim(int K) {
    switch (K) {
      case 6: return detail::cublasdx_blockdim_k<T, 6>();
      case 8: return detail::cublasdx_blockdim_k<T, 8>();
      case 10: return detail::cublasdx_blockdim_k<T, 10>();
      case 12: return detail::cublasdx_blockdim_k<T, 12>();
      case 16: return detail::cublasdx_blockdim_k<T, 16>();
      case 20: return detail::cublasdx_blockdim_k<T, 20>();
      case 32: return detail::cublasdx_blockdim_k<T, 32>();
      default: THROW("CUBLASdx: Unsupported K");
    }
  }

} // namespace mra

#else

#error "cublasdx not found, please install cublasdx and set CUDADIR to the correct path"

#endif // __has_include(<cublasdx.hpp>)

#endif // MRA_OPS_MXM_CUBLASDX_H
