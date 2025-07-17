#ifndef HAVE_TRANSFORM_CUBLASDX_H
#define HAVE_TRANSFORM_CUBLASDX_H

#include "util.h"
#include "mxm_cublasdx.h"


/**********************************************************************************
 * cublasDx implementation of transform
 *
 * The cublasDx implementation uses the cublasdx library directly to perform
 * the tensor transformation. It relies on a single shared memory tensor
 * to which the result of a GEMM is written in each iteration.
 * The register fragment saves us the additional shared memory tensor
 * that would be required to store the result of the GEMM.
 **********************************************************************************/

template <typename T, int K>
__forceinline__ __device__
void transform_cublasdx_k(
    const T* t, // input tensor
    const T* c, // input matrix
    T* result)
{
  constexpr const int ndim = 3; // fixed for benchmark
  using GEMM = typename mra::detail::GEMMBuilder<T, K*K, K, K,
                                                cublasdx::col_major,
                                                cublasdx::row_major,
                                                cublasdx::row_major>::GEMM;

  using alignment = cublasdx::alignment_of<GEMM>;

  extern __shared__ __align__(16) char smem[];

  auto [smem_a, smem_b] =
    cublasdx::shared_memory::slice_into_pointers<GEMM::a_value_type, GEMM::b_value_type>(
        smem,
        cublasdx::alignment_of_v_a<GEMM>, cublasdx::cosize(GET_SHARED_LAYOUT(GEMM, a)),
        cublasdx::alignment_of_v_b<GEMM>, cublasdx::cosize(GET_SHARED_LAYOUT(GEMM, b)));


  /* global memory tensors */
  auto a_global_tensor = cublasdx::make_tensor(t, GEMM::get_layout_gmem_a());
  auto b_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_b());
  auto c_global_tensor = cublasdx::make_tensor(result, GEMM::get_layout_gmem_c());

  /* shared memory tensors */
  auto a_shared_tensor   = cublasdx::make_tensor(smem_a,   GET_SHARED_LAYOUT(GEMM, a));
  auto b_shared_tensor   = cublasdx::make_tensor(smem_b,   GET_SHARED_LAYOUT(GEMM, b));

  cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
  cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);

  /* wait for loads to complete */
  cublasdx::copy_wait();

  for (int n=0; n<ndim; ++n) {
    /* execute the GEMM operation */
    auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

    /* wait for all threads to complete so we can write the result back to shared memory */
    __syncthreads();

    /* copy the result over to the the A shared tensor */
    cublasdx::copy_fragment<alignment::a>(c_register_fragment, a_shared_tensor, partitioner);

    /* wait for stores to complete */
    cublasdx::copy_wait();
  }
  /* copy the result from shared memory to global memory */
  cublasdx::copy<GEMM, alignment::c>(a_shared_tensor, c_global_tensor);

  /* wait for the copy to complete */
  cublasdx::copy_wait();
}


template <typename T, int K>
__forceinline__ __device__
void transform_cublasdx(
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  return transform_cublasdx_k<T, K>(t, c, result);

#if 0
  (void)workspace; // unused in this implementation
  switch (K) {
    case 8 : transform_cublasdx_k<T,  8>(t, c, result); break;
    case 10: transform_cublasdx_k<T, 10>(t, c, result); break;
    case 16: transform_cublasdx_k<T, 16>(t, c, result); break;
    case 20: transform_cublasdx_k<T, 20>(t, c, result); break;
    default:
      printf("Unsupported K value: %d\n", K);
      return;
  }
#endif // 0
  /* no need to synchronize here, cublasdx::copy_wait() synchronizes */
}

template <typename T, int K>
constexpr int transform_cublasdx_shmem_size_k()
{
  using GEMM = typename mra::detail::GEMMBuilder<T, K*K, K, K,
                                                cublasdx::col_major,
                                                cublasdx::row_major,
                                                cublasdx::row_major>::GEMM;

  auto calc = cublasdx::make_shared_storage_calculator()
                  .add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GET_SHARED_LAYOUT(GEMM, a))
                  .add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GET_SHARED_LAYOUT(GEMM, b));
  auto smem_size = calc.get();
  return smem_size;
}

template <typename T>
int transform_cublasdx_shmem_size(int K)
{
  switch (K) {
    case 8 : return transform_cublasdx_shmem_size_k<T,  8>();
    case 10: return transform_cublasdx_shmem_size_k<T, 10>();
    case 16: return transform_cublasdx_shmem_size_k<T, 16>();
    case 20: return transform_cublasdx_shmem_size_k<T, 20>();
    default:
      printf("Unsupported K value: %d\n", K);
      return 0;
  }
  /* no need to synchronize here, cublasdx::copy_wait() synchronizes */
}


template <typename T, int K>
constexpr auto transform_cublasdx_block_dim()
{
  using GEMM = typename mra::detail::GEMMBuilder<T, K*K, K, K,
                                                cublasdx::col_major,
                                                cublasdx::row_major,
                                                cublasdx::row_major>::GEMM;

  return GEMM::suggested_block_dim;
}


template <typename T, int K>
constexpr auto transform_cublasdx_block_size()
{
  auto blockdims = transform_cublasdx_block_dim<T, K>();
  return blockdims.x * blockdims.y * blockdims.z;
}


template<typename T, int K>
LAUNCH_BOUNDS((transform_cublasdx_block_size<T, K>()), 1)
__global__ void transform_cublasdx_kernel(int nfuncs, const T* A, const T* B, T* C, T* workspace) {

  const T *a, *b;
  T *c, *w;
  int K2NDIM = K*K*K;
  /* workspace is allocated for each thread-block */
  w = workspace + blockIdx.x * K2NDIM;
  /* iterate over all tensors */
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    a = A + i * K2NDIM;
    b = B;
    c = C + i * K2NDIM;
    transform_cublasdx<T, K>(a, b, c, w);
  }
}

template<typename T>
void submit_transform_cublasdx_bench(int nfuncs, int nblocks, int K,
                                    const T* A, const T* B, T* C, T* workspace,
                                    cudaStream_t stream)
{
  auto smem_size = transform_cublasdx_shmem_size<T>(K);
  switch (K) {
  case 8: {
    CONFIGURE_KERNEL((transform_cublasdx_kernel<T, 8>), smem_size);
    CALL_KERNEL((transform_cublasdx_kernel<T, 8>), std::min(nfuncs, nblocks), (transform_cublasdx_block_dim<T, 8>()), smem_size, stream, (nfuncs, A, B, C, workspace));
    break;
  }
  case 10: {
    CONFIGURE_KERNEL((transform_cublasdx_kernel<T, 10>), smem_size);
    CALL_KERNEL((transform_cublasdx_kernel<T, 10>), std::min(nfuncs, nblocks), (transform_cublasdx_block_dim<T, 10>()), smem_size, stream, (nfuncs, A, B, C, workspace));
    break;
  }
  case 16: {
    CONFIGURE_KERNEL((transform_cublasdx_kernel<T, 16>), smem_size);
    CALL_KERNEL((transform_cublasdx_kernel<T, 16>), std::min(nfuncs, nblocks), (transform_cublasdx_block_dim<T, 16>()), smem_size, stream, (nfuncs, A, B, C, workspace));
    break;
  }
  case 20: {
    CONFIGURE_KERNEL((transform_cublasdx_kernel<T, 20>), smem_size);
    CALL_KERNEL((transform_cublasdx_kernel<T, 20>), std::min(nfuncs, nblocks), (transform_cublasdx_block_dim<T, 20>()), smem_size, stream, (nfuncs, A, B, C, workspace));
    break;
  }
  default:
    printf("Unsupported K value: %d\n", K);
    throw std::runtime_error("Unsupported K value in transform_cublasdx_bench");
  }
}


#endif // HAVE_TRANSFORM_CUBLASDX_H