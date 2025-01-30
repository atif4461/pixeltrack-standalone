#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/ScopedSetDevice.h"

#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize = allocator::intPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
/**
 * Allocates host memory of specified size with optional caching policy 
 */
// The above comment was written by an LLM. 
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ptr = allocator::getCachingHostAllocator().allocate(nbytes, stream);
    } else {
      cudaCheck(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Allocates host memory of specified size in bytes on the given device 
 * associated with the provided CUDA stream.
 *
 * @param device   The device on which to allocate the memory
 * @param nbytes   The number of bytes to allocate
 * @param stream   The CUDA stream associated with the allocation
 * @return A pointer to the allocated memory
 */
// The above comment was written by an LLM. 
  void *allocate_host(int device, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ScopedSetDevice setDevice(device);
      ptr = allocator::getCachingHostAllocator().allocate(nbytes, stream);
    } else {
      cudaCheck(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees host memory allocated by the system
 * @param ptr pointer to the memory block to be freed
 */
// The above comment was written by an LLM. 
  void free_host(void *ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingHostAllocator().free(ptr);
    } else {
      cudaCheck(cudaFreeHost(ptr));
    }
  }

}  // namespace cms::cuda
