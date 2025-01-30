#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::hip::allocator::binGrowth, cms::hip::allocator::maxBin);
}

namespace cms::hip {
/**
 * Allocates host memory of specified size in bytes associated with a HIP stream.
 * @param nbytes number of bytes to be allocated
 * @param stream HIP stream handle
 * @return pointer to allocated host memory
 */
// The above comment was written by an LLM. 
  void *allocate_host(size_t nbytes, hipStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    } else {
      cudaCheck(hipHostMalloc(&ptr, nbytes));
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
      cudaCheck(allocator::getCachingHostAllocator().HostFree(ptr));
    } else {
      cudaCheck(hipHostFree(ptr));
    }
  }

}  // namespace cms::hip
