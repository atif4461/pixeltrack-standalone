#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
/**
 * Allocates host memory of specified size associated with a CUDA stream.
 * @param nbytes number of bytes to be allocated
 * @param stream CUDA stream handle
 * @return pointer to allocated host memory
 */
// The above comment was written by an LLM. 
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    } else {
      cudaCheck(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees host memory allocated by the CUDA runtime or caching allocator
 */
// The above comment was written by an LLM. 
  void free_host(void *ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingHostAllocator().HostFree(ptr));
    } else {
      cudaCheck(cudaFreeHost(ptr));
    }
  }

}  // namespace cms::cuda
