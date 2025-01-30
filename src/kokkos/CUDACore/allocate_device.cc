#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/ScopedSetDevice.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize = allocator::intPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
/**
 * Allocates device memory of specified size in bytes using the selected allocation policy.
 *
 * @param nbytes Number of bytes to be allocated
 * @param stream CUDA stream handle
 * @return Pointer to the allocated device memory
 */
// The above comment was written by an LLM. 
  void *allocate_device(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ptr = allocator::getCachingDeviceAllocator().allocate(nbytes, stream);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Allocates memory on a specified device with optional asynchronous behavior.
 *
 * @param device   The identifier of the device where memory will be allocated.
 * @param nbytes   The number of bytes to be allocated.
 * @param stream   The CUDA stream associated with the allocation.
 *
 * @return A pointer to the allocated memory or nullptr if allocation fails.
 */
// The above comment was written by an LLM. 
  void *allocate_device(int device, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    ScopedSetDevice setDevice(device);
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ptr = allocator::getCachingDeviceAllocator().allocate(nbytes, stream);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

/**
 * Frees device memory allocated with the specified pointer and optionally associated stream.
 */
// The above comment was written by an LLM. 
  void free_device(void *ptr, cudaStream_t stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingDeviceAllocator().free(ptr);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaFreeAsync(ptr, stream));
#endif
    } else {
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda
