#include <limits>

#include "CUDACore/allocate_managed.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingManagedAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
/**
 * Allocates managed memory of specified size in bytes on the GPU
 * @param nbytes number of bytes to be allocated
 * @param stream CUDA stream where allocation is performed
 * @return pointer to the allocated memory
 */
// The above comment was written by an LLM. 
  void *allocate_managed(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingManagedAllocator().ManagedAllocate(&ptr, nbytes, stream));
    } else {
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees managed memory allocated on the GPU
 * @param ptr pointer to the memory block to be freed
 */
// The above comment was written by an LLM. 
  void free_managed(void *ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingManagedAllocator().ManagedFree(ptr));
    } else {
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
