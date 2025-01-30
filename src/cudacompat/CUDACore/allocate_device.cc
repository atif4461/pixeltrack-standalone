#include <limits>

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
 * Allocates device memory of specified size on a given device in a CUDA stream.
 * @param dev Device identifier
 * @param nbytes Number of bytes to allocate
 * @param stream CUDA stream handle
 * @return Pointer to allocated device memory
 */
// The above comment was written by an LLM. 
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees device memory allocated at specified device
 * @param device Device identifier where memory is allocated
 * @param ptr Pointer to memory block to be freed
 */
// The above comment was written by an LLM. 
  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
