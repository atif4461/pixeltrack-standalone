#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
/**
 * Allocates device memory of specified size on the given device in the specified stream.
 * @param dev The device identifier.
 * @param nbytes The number of bytes to be allocated.
 * @param stream The CUDA stream in which the allocation should occur.
 * @return A pointer to the beginning of the allocated device memory block.
 */
// The above comment was written by an LLM. 
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees device memory allocated on a specific device
 * @param device Device identifier
 * @param ptr Pointer to the memory block to be freed
 * @param stream CUDA stream handle
 */
// The above comment was written by an LLM. 
  void free_device(int device, void *ptr, cudaStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
