#pragma once
#include <assert.h>


extern __shared__ char arena[];

// ---- Alignment helpers -----
__device__ __forceinline__
constexpr size_t align_up(size_t offset, size_t align) {
    return (offset + align - 1u) & ~(align - 1u);
}

class BlockStackAllocator {
public:
    static constexpr size_t DefaultAlign = 16;

    template<typename T>
    class BlockScopedAlloc {
    public:
        __device__ BlockScopedAlloc(BlockStackAllocator& bsa, T* ptr, size_t cp)
            : bsa_(bsa), ptr_(ptr), cp_(cp) {}

        BlockScopedAlloc(const BlockScopedAlloc&)            = delete;
        BlockScopedAlloc& operator=(const BlockScopedAlloc&) = delete;

        __device__ BlockScopedAlloc(BlockScopedAlloc&& o)
            : bsa_(o.bsa_), ptr_(o.ptr_), cp_(o.cp_) {
            o.ptr_ = nullptr;
        }

        __device__ ~BlockScopedAlloc() {
            if (ptr_) bsa_.restore(cp_);
        }

        __device__ operator T*() const { return ptr_; }
        __device__ T* get()      const { return ptr_; }

    private:
        BlockStackAllocator& bsa_;
        T*     ptr_;
        size_t cp_;
    };

    __device__ void init(size_t capacity) {
        if (threadIdx.x == 0) {
            capacity_ = capacity;
            offset_   = 0;
        }
        __syncthreads();
    }

    __device__ void* alloc_raw(size_t bytes, size_t align = DefaultAlign) {
        __shared__ size_t alloc_start;
        if (threadIdx.x == 0) {
            size_t aligned = align_up(offset_, align);
            if (aligned + bytes <= capacity_) {
                alloc_start = aligned;
                offset_     = aligned + bytes;
            } else {
                alloc_start = capacity_ + 1; // sentinel: out of memory
            }
        }
        __syncthreads();
        if (alloc_start > capacity_) return nullptr;
        return static_cast<void*>(arena + alloc_start);
    }

    template<typename T>
    __device__ BlockScopedAlloc<T> alloc(size_t count = 1, bool zero_init = false) {
        size_t cp = offset_;
        __syncthreads();

        T* ptr = static_cast<T*>(alloc_raw(sizeof(T) * count, alignof(T)));

        return BlockScopedAlloc<T>(*this, ptr, cp);
    }

    // ---- Collective checkpoint / restore / reset -----------------------
    __device__ size_t checkpoint() {
        __syncthreads();
        return offset_;
    }

    __device__ void restore(size_t cp) {
        if (threadIdx.x == 0) offset_ = cp;
        __syncthreads();
    }

    __device__ void reset() {
        if (threadIdx.x == 0) offset_ = 0;
        __syncthreads();
    }

    // ---- Queries -------------------------------------------------------
    __device__ size_t used()      const { return offset_; }
    __device__ size_t remaining() const { return capacity_ - offset_; }
    __device__ size_t capacity()  const { return capacity_; }

private:
    size_t capacity_;
    size_t offset_;
};


__device__ __forceinline__
BlockStackAllocator& block_allocator() {
    static __shared__ BlockStackAllocator bsa;
    return bsa;
}
