#include <cassert>
#include <limits>

#include <hip/hip_runtime.h>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::hip::allocator::binGrowth, cms::hip::allocator::maxBin);
}

namespace cms::hip {
/**
 * Allocates device memory of specified size on the given device in the specified stream.
 * @param dev The device identifier.
 * @param nbytes The number of bytes to allocate.
 * @param stream The stream in which the allocation should occur.
 * @return A pointer to the allocated device memory.
 */
// The above comment was written by an LLM. 
  void *allocate_device(int dev, size_t nbytes, hipStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
#if HIP_VERSION >= 50200000
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(hipMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(hipMalloc(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees device memory allocated on the specified device
 * @param device Device from which to free the memory
 * @param ptr Pointer to the memory block to be freed
 * @param stream Stream on which to perform the free operation
 */
// The above comment was written by an LLM. 
  void free_device(int device, void *ptr, hipStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
#if HIP_VERSION >= 50200000
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(hipFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(hipFree(ptr));
    }
  }

}  // namespace cms::hip
