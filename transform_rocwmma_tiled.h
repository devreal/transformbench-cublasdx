#ifndef HAVE_TRANSFORM_ROCWMMA_TILED_H
#define HAVE_TRANSFORM_ROCWMMA_TILED_H

#include "util.h"
#include "mxm.h"
#include "transform_rocwmma.h"

#define ROCWMMA_NUM_THREADS 64
#define ROCWMMA_TILE_K 4 // tiling factor along K dimension for K >= 16

template<size_type K, typename T>
struct RocWMMAConfig {
  static constexpr int get_mn() {
      if (K == 16) return 16; // used for K = 16
      else return 4; // use 4 for K = 20 & 24
  };
  static constexpr uint32_t WAVE = 64;   // CDNA wavefront size
  static constexpr int M = K*K;
  static constexpr int N = K;
  static constexpr int TileM = get_mn(); // elements per tile in M dimension
  static constexpr int TileN = TileM; // square tiles
  static constexpr int TileK = ROCWMMA_TILE_K; // tiling along the reduction dimension K, to increase reuse of A and B fragments
  static constexpr int FragsK = K / TileK; // number of tiles along K dimension
  static constexpr int FragsM = M / TileM; // number of tiles along M dimension
  static constexpr int FragsN = N / TileN; // number of tiles along N dimension
  static constexpr int NumWaves = ROCWMMA_NUM_THREADS / WAVE;
  static constexpr int FragsPerWaveM = FragsM / NumWaves; // number of M tiles (output tiles) computed per wavefront
  static constexpr int FragsPerWaveN = FragsN; // number of N tiles computed per wavefront
  static constexpr int SuperBlocksM = FragsK; // number of super-blocks along M dimension, same as K tiling
  static constexpr int FragsPerSuperBlockM = FragsM / SuperBlocksM; // number of M tiles in a super-block
  static constexpr int FragsPerWaveSuperBlockM = FragsPerWaveM / SuperBlocksM; // number of M tiles in a super-block

  // sanity check the parameters
  static_assert(FragsM % NumWaves == 0, "FragsM must be divisible by number of waves");
  static_assert(FragsPerWaveSuperBlockM * SuperBlocksM == FragsPerWaveM);
  static_assert(FragsK == SuperBlocksM, "K-Tiling must match Super Block Tiling!");
};


#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

/**
 * This implementation is for K 16, 20, 24. We split the K dimension into tiles of size 4 or 8.
 * We split the K dimension into smaller tiles. The number of K tiles determines the number of superblocks
 * in the M dimension, which determines the reuse of A and B fragments in shared memory.
 * For example, if we split K=16 into tiles of 4, we have 4 K tiles.
 * We can then split the M dimension into 4 superblocks.
 * Once we computed a superblock of C, we write it to shared memory.
 * Once we consumed the first K/4 columns of A, we can read the superblock of C from shared memory
 * into registers for the K/4 columns of the next A.
 * We can start computing the first partial contribution for the first superblock of the next C
 * while we compute the next superblock.
 */
template <size_type K, typename T>
__device__ void transform_rocwmma_tiled_k(
    const T* a,
    const T* b,
    T*& c,
    T* workspace)
{
  constexpr const int ndim = 1; // fixed for benchmark

  if constexpr (K < 16) {
    // Fallback to non mma implementation
    transform_klt16<K, T>(a, b, c);
    return;
  } else if constexpr (K > 16) {
    // Not supported, fallback to Level-3
    transform_level3_k<T, K>(a, b, c, workspace);
    return;
  } else {

    using Config = RocWMMAConfig<K,T>;
    constexpr uint32_t WAVE = Config::WAVE;   // CDNA wavefront size
    constexpr int M = Config::M;
    constexpr int N = Config::N;
    constexpr int TileM = Config::TileM; // elements per tile in M dimension
    constexpr int TileN = Config::TileN; // square tiles
    constexpr int TileK = Config::TileK; // tile by 4 along the reduction dimension K, to increase reuse of A and B fragments
    constexpr int FragsK = Config::FragsK; // number of tiles along K dimension
    constexpr int FragsM = Config::FragsM; // number of tiles along M dimension
    constexpr int FragsN = Config::FragsN; // number of tiles along N dimension
    constexpr int NumWaves = Config::NumWaves; // number of waves
    constexpr int FragsPerWaveM = Config::FragsPerWaveM; // number of M tiles (output tiles) computed per wavefront
    constexpr int FragsPerWaveN = Config::FragsPerWaveN; // number of N tiles computed per wavefront
    constexpr int SuperBlocksM = Config::SuperBlocksM; // number of super-blocks along M dimension
    constexpr int FragsPerSuperBlockM = Config::FragsPerSuperBlockM; // number of M tiles in a super-block
    constexpr int FragsPerWaveSuperBlockM = Config::FragsPerWaveSuperBlockM; // number of M tiles in a super-block

    using FragmentA = rocwmma::fragment<rocwmma::matrix_a, TileM, TileN, TileK, T, rocwmma::col_major>;
    using FragmentB = rocwmma::fragment<rocwmma::matrix_b, TileM, TileN, TileK, T, rocwmma::row_major>;
    using FragmentAcc = rocwmma::fragment<rocwmma::accumulator, TileM, TileN, TileK, T, rocwmma::row_major>;

    FragmentB b_frags[FragsK][FragsN];
    FragmentA a_frags[SuperBlocksM][FragsPerWaveSuperBlockM][FragsK];
    FragmentAcc acc_frags[SuperBlocksM][FragsPerWaveSuperBlockM][FragsPerWaveN];

    /* single shared memory region, holds A and C */
    extern __shared__ T shmem[];

    int wave_id = thread_id() / WAVE;

    // returns the offset in A in global or shared memory.
    // for shared memory, use sb = 0
    // takes the super block index, the wave-local fragment index in M dimension, and the K tile index
    auto a_frag_offset = [&](int sb, int local_m, int k) {
      // load from memory in [k, m] order, with layout [K, K^2]
      if (thread_id() == 0) printf("a_frag_offset(%d, %d, %d) -> %d\n", sb, local_m, k, ((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM + k * K*K));
      return ((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM + k * K*K);
    };

    auto b_frag_offset = [&](int k, int n) {
      // load from memory in [k, n] order, with layout [K, K]
      return (k * TileK * K + n * TileN);
    };

    // returns the offset in C in global or shared memory.
    // for shared memory, use sb = 0
    auto c_frag_offset = [&](int sb, int local_m, int n) {
      // store to memory in [m, n] order, with layout [K^2, K]
      if (thread_id() == 0) printf("c_frag_offset(%d, %d, %d) -> %d\n", sb, local_m, n, ((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM * N + n * TileN));
      return ((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM * N + n * TileN);
    };

    // load all b into wave 0 registers and store them back to shared memory
    // loading to LDS goes through registers so we might as well keep them at wave 0
    for (int i = thread_id(); i < (2*K*K*ROCWMMA_TILE_K); i += block_size()) shmem[i] = -1111111;
    rocwmma::synchronize_workgroup();
    if (wave_id == 0) {
      for (int k = 0; k < FragsK; ++k) {
        for (int n = 0; n < FragsN; ++n) {
          rocwmma::load_matrix_sync(b_frags[k][n], b + b_frag_offset(k, n), K);
        }
      }
      for (int k = 0; k < FragsK; ++k) {
        for (int n = 0; n < FragsN; ++n) {
          rocwmma::store_matrix_sync(shmem + b_frag_offset(k, n), b_frags[k][n], K); // store B in shared memory for reuse
        }
      }
    }
    // make sure all b fragments are in shared memory before we load them into other waves' registers
    rocwmma::synchronize_workgroup();
    // now have all other waves load b from shared memory into registers
    if (wave_id > 0) {
      // load B fragments from shared memory
      for (int k = 0; k < FragsK; ++k) {
        for (int n = 0; n < FragsN; ++n) {
          rocwmma::load_matrix_sync(b_frags[k][n], shmem + b_frag_offset(k, n), K);
        }
      }
    }

    // start loading all of A into registers
    for (int sb = 0; sb < SuperBlocksM; ++sb) {
      for (int m = 0; m < FragsPerWaveSuperBlockM; ++m) {
        for (int k = 0; k < FragsK; ++k) {
          //const T* a_ptr = a + a_frag_offset(sb, m, k);
	  // ((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM + k * K*K)
	  //((sb * FragsPerSuperBlockM + wave_id + local_m*NumWaves) * TileM + k * K*K)
	  //const T* a_ptr = a + (sb*FragsPerSuperBlockM + wave_id + local_m*NumWaves)*TileM + k*K;
	  //const T* a_ptr = a + c_frag_offset(sb, m, k);
	  const T* a_ptr = a + ((sb * FragsPerSuperBlockM + wave_id + m*NumWaves) * TileM + k * K*K);
          rocwmma::load_matrix_sync(a_frags[sb][m][k], a_ptr, K*K);
        }
      }
    }


    // wait for all waves to finish writing to global memory before returning
    rocwmma::synchronize_workgroup();
      for (int sb = 0; sb < SuperBlocksM; ++sb) {
        for (int i = 0; i < FragsPerWaveSuperBlockM; ++i) {
          for (int j = 0; j < FragsK; ++j) {
	   T *a_ptr = shmem + ((sb * FragsPerSuperBlockM + wave_id + i*NumWaves) * TileM * N + j*TileK);
	   if (thread_id() == 0) printf("Writing A tile sb %d m %d n %d to %d", a_ptr - shmem);
	   rocwmma::store_matrix_sync(a_ptr,
                                      a_frags[sb][i][j], K); 
          }
        }
        rocwmma::synchronize_workgroup();
    	if (thread_id() == 0) {
		printf("A sb %d\n", sb);
        	for (int i = 0; i < K*K/ROCWMMA_TILE_K; ++i) {
          		for (int j = 0; j < K; ++j) {
				printf(" %6.1f", shmem[i*K+j]);
            		//printf("INITIAL B, tile k=%d, n=%d, b[0] = %f\n", i, j, b_frags[i][j].x[0]);
          		}
			printf("\n");
        	}
      	}
        rocwmma::synchronize_workgroup();
      }
    // wait for all waves to finish writing to global memory before returning
    rocwmma::synchronize_workgroup();


    // main loop: iterate over dimensions
    for (int d = 0; d < ndim; ++d) {
      T *c_ptr = (d == ndim-1) ? c : shmem; // write to global memory if it's the last iteration, otherwise write to shared memory for the next iteration
      // iterate over super-blocks in M dimension
      for (int sb = 0; sb < SuperBlocksM; ++sb) {
        // compute the whole super-block

        // wait for all waves to finish writing to global memory before returning
        rocwmma::synchronize_workgroup();

        if (thread_id() == 0) {
	  printf("d = %d, sb = %d\n", d, sb);

          for (int sb = 0; sb < SuperBlocksM; ++sb) {
            for (int i = 0; i < FragsPerWaveSuperBlockM; ++i) {
              for (int j = 0; j < FragsK; ++j) {
                printf("super-block %d, d %d, tile m=%d, k=%d, a[0] = %f\n", sb, d, i, j, a_frags[sb][i][j].x[0]);
		T val = a_frags[sb][i][j].x[0];
		for (int k = 0; k < TileM*TileK; ++k) {
		  if (val != a_frags[sb][i][j].x[k]) {
                  printf("DIFFERENT element at index %d: %f\n", k, a_frags[sb][i][j].x[k]);
		  val = a_frags[sb][i][j].x[k];
		  }
                }
              }
            }
          }
        for (int i = 0; i < FragsK; ++i) {
          for (int j = 0; j < FragsN; ++j) {
            printf("B, tile k=%d, n=%d, b[0] = %f\n", i, j, b_frags[i][j].x[0]);
	    T val = b_frags[i][j].x[0];
	    for (int k = 0; k < TileM*TileN; ++k) {
		if (val != b_frags[i][j].x[k]) {
			printf("DIFFERENT element at index %d: %f\n", k, b_frags[i][j].x[k]);
			val = b_frags[i][j].x[k];
		}
	    }
          }
        }


        }
        rocwmma::synchronize_workgroup();


        for (int m = 0; m < FragsPerWaveSuperBlockM; ++m) {
          for (int n = 0; n < FragsPerWaveN; ++n) {
            auto& acc = acc_frags[sb][m][n];
            if (sb == 0) { // column 0
              // initialize to 0 only for the first super block
              // all other super blocks will already have partials sums from columns 0..sb-1 of A
              rocwmma::fill_fragment(acc, static_cast<T>(0));
            }
            // we have already computed with columns 0..sb-1 of A and loaded them back from shared memory for the next iteration
            // so start at an offset
            for (int k = sb; k < FragsK; ++k) {

            rocwmma::synchronize_workgroup();
	    if (thread_id() == 0) printf("MMA d %d sb %d m %d n %d k %d\n", d, sb, m, n, k);

	    auto& a_frag = a_frags[sb][m][k];
	    auto& b_frag = b_frags[k][n];
            if (thread_id() == 0) {
              printf("MMA BEFORE sb %d, d %d, tile m=%d, n=%d\n", sb, d, m, n);
              printf("A: ");
              for (int kk = 0; kk < TileM*TileK; ++kk) {
		if (kk % TileK == 0) printf("\n");
                printf(" %5.1f", a_frag.x[kk]);
              }
              printf("\nB: ");
              for (int kk = 0; kk < TileK*TileN; ++kk) {
		if (kk % TileN == 0) printf("\n");
                printf(" %5.1f", b_frag.x[kk]);
              }

              printf("\nACC: ");
              for (int kk = 0; kk < TileM*TileN; ++kk) {
		if (kk % TileN == 0) printf("\n");
                printf(" %5.1f", acc.x[kk]);
              }
              printf("\n");
            }
            rocwmma::synchronize_workgroup();
	    FragmentAcc frag;
	    rocwmma::fill_fragment(frag, static_cast<T>(0));
              rocwmma::mma_sync(acc, a_frag, b_frag, acc);
            rocwmma::synchronize_workgroup();
	    if (thread_id() == 0) {
              printf("MMA AFTER sb %d, d %d, tile m=%d, n=%d\n", sb, d, m, n);
              printf("\nACC: ");
              for (int kk = 0; kk < TileM*TileN; ++kk) {
		if (kk % TileN == 0) printf("\n");
                printf(" %5.1f", acc.x[kk]);
              }
              printf("\n");
            }
	    //acc = frag;

            rocwmma::synchronize_workgroup();

            }
          }
        }

        // we have computed a super-block of C
        // wait for every wave to have read either the B tiles (first iteration)
        // or the A tiles (subsequent iterations) before we write to shared memory
        // can skip this for the last dimension since we write to global memory and there is no reuse after this
        //if (d < ndim-1) {
          rocwmma::synchronize_workgroup();
        //}
        // write the super block back to shared memory so we can "rotate" and read in again
        for (int m = 0; m < FragsPerWaveSuperBlockM; ++m) {
          for (int n = 0; n < FragsPerWaveN; ++n) {
            // write order [m, n] in shared memory, with layout [M, N]
            //rocwmma::store_matrix_sync(c_ptr + c_frag_offset((d < ndim-1) ? 0 : sb, m, n),
            rocwmma::store_matrix_sync(shmem + c_frag_offset(0, m, n),
                                       acc_frags[sb][m][n], K);
          }
        }

	rocwmma::synchronize_workgroup();

	for (int i = thread_id(); i < K*K*K/ROCWMMA_TILE_K; i += block_size()) c[sb*K*K*K/ROCWMMA_TILE_K + i] = shmem[i];
	rocwmma::synchronize_workgroup();

        // while writing to shared memory, we compute the partials using the tiles in column sb of A
        // TODO: not sure whether there is any actual overlap here
        for (int sbx = sb+1; sbx < SuperBlocksM; ++sbx) {
          for (int m = 0; m < FragsPerWaveSuperBlockM; ++m) {
            for (int n = 0; n < FragsPerWaveN; ++n) {
              auto& acc = acc_frags[sbx][m][n];
              if (sb == 0) { // column 0
                // first time we compute this super-block, so initialize the accumulator to 0
                rocwmma::fill_fragment(acc, static_cast<T>(0));
              }
              rocwmma::mma_sync(acc, a_frags[sbx][m][sb], b_frags[sb][n], acc);
            }
          }
        }

          rocwmma::synchronize_workgroup();
        if (thread_id() == 0) {

          for (int sb = 0; sb < SuperBlocksM; ++sb) {
            for (int i = 0; i < FragsPerWaveSuperBlockM; ++i) {
              for (int j = 0; j < FragsN; ++j) {
                printf("super-block %d, d %d, tile m=%d, n=%d, acc[0] = %f\n", sb, d, i, j, acc_frags[sb][i][j].x[0]);
		T val = acc_frags[sb][i][j].x[0];
		for (int k = 0; k < TileM*TileN; ++k) {
		  if (val != acc_frags[sb][i][j].x[k]) {
                  printf("DIFFERENT element at index %d: %f\n", k, acc_frags[sb][i][j].x[k]);
		  val = acc_frags[sb][i][j].x[k];
		  }
                }
              }
            }
          }
	  printf("C after sb %d shmem\n", sb);
	  for (int m = 0; m < K*K/ROCWMMA_TILE_K; ++m) {
		  for (int n = 0; n < K; ++n) 
			  printf(" %6.1f", shmem[m*K + n]);
		  printf("\n");
	  }

	  printf("C after sb %d\n", sb);
	  for (int m = 0; m < K*K; ++m) {
		  for (int n = 0; n < K; ++n) 
			  printf(" %6.1f", c[m*K*K + n]);
		  printf("\n");
	  }

	}
        if (d < ndim-1) {
          // wait for all waves to finish writing the super-block before we read it back in the next iteration
          rocwmma::synchronize_workgroup();
          // read the tiles in column sb back into registers for the next iteration
          for (int sbx = 0; sbx < SuperBlocksM; ++sbx) {
            for (int m = 0; m < FragsPerWaveSuperBlockM; ++m) {
              T* a_ptr = shmem + a_frag_offset(sbx, m, 0);
              //T* a_ptr = shmem + a_frag_offset(0, m, sb);
              rocwmma::load_matrix_sync(a_frags[sbx][m][sb], a_ptr, K*K); // load the super-block of C back from shared memory for the next iteration
            }
          }
        }
        // wait for all waves to finish writing to global memory before returning
        rocwmma::synchronize_workgroup();
      }
    }

    // wait for all waves to finish writing to global memory before returning
    rocwmma::synchronize_workgroup();

  }
}

#endif // not __HIP_DEVICE_COMPILE__

// fwd-decl for kernel
template <size_type K, typename T>
__device__ void transform_rocwmma_tiled_k(
    const T* a,
    const T* b,
    T*& c,
    T* workspace);

/* One kernel binary per K — register pressure is proportional to K, not max(K). */
template <typename T, int K>
LAUNCH_BOUNDS(ROCWMMA_NUM_THREADS, 1)
__global__ void transform_rocwmma_tiled(int nfuncs,
                                        const T* A, const T* B, T* C, T* workspace) {
  constexpr int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    /* result pointer starts at c; workspace is w */
    T* result  = c;
    transform_rocwmma_tiled_k<K>(a, B, result, w);
  }
}

template <typename T>
inline int transform_rocwmma_tiled_shmem_size(int K) {
  if (K < 16) {
    // For K<=16, we load A and B into shared memory. We need space for A (K^3), B (K^2), and C (K^3).
    return (K*K*K + K*K) * sizeof(T);
  } else if (K == 16) {
    // For K==16, we hold one copy of A/C in LDS
    return (2*K*K*ROCWMMA_TILE_K) * sizeof(T);
  } else {
    return transform_level3_shmem_size<T>(K);
  }
}

template <typename T>
inline Dim3 transform_rocwmma_tiled_blockdim(int K) {
  return {ROCWMMA_NUM_THREADS, 1, 1};
}

template <typename T>
inline void submit_transform_rocwmma_tiled_bench(int nfuncs, int nblocks, int K,
                                                  const T* A, const T* B, T* C, T* workspace,
                                                  Stream stream)
{
  Dim3 thread_dims = transform_rocwmma_tiled_blockdim<T>(K);
  int  smem_size   = transform_rocwmma_tiled_shmem_size<T>(K);

#define DISPATCH_L3(Kval) \
  case Kval: \
    CONFIGURE_KERNEL((transform_rocwmma_tiled<T, Kval>), smem_size); \
    CALL_KERNEL((transform_rocwmma_tiled<T, Kval>), std::min(nfuncs, nblocks), \
                thread_dims, smem_size, stream, \
                (nfuncs, A, B, C, workspace)); \
    break;

  switch (K) {
    DISPATCH_L3( 6)
    DISPATCH_L3( 8)
    DISPATCH_L3(10)
    DISPATCH_L3(12)
    DISPATCH_L3(16)
    DISPATCH_L3(20)
    DISPATCH_L3(24)
    default:
      printf("submit_transform_rocwmma_tiled_bench: unsupported K=%d\n", K);
  }
#undef DISPATCH_L3
}


#undef ROCWMMA_NUM_THREADS
#undef ROCWMMA_TILE_K

#endif // HAVE_TRANSFORM_ROCWMMA_TILED_H
