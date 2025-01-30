#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::cuda {
/**
 * Destroys a CUDA stream object and sets the device accordingly.
 */
// The above comment was written by an LLM. 
  void StreamCache::Deleter::operator()(cudaStream_t stream) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      cudaCheck(cudaStreamDestroy(stream));
    }
  }

  // StreamCache should be constructed by the first call to
  // getStreamCache() only if we have CUDA devices present
  StreamCache::StreamCache() : cache_(deviceCount()) {}

/**
 * Retrieves a shared pointer to a CUDA stream object from the cache,
 * creating it if necessary, with non-blocking behavior enabled. 
 */
// The above comment was written by an LLM. 
  SharedStreamPtr StreamCache::get() {
    const auto dev = currentDevice();
    return cache_[dev].makeOrGet([dev]() {
      cudaStream_t stream;
      cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      return std::unique_ptr<BareStream, Deleter>(stream, Deleter{dev});
    });
  }

/**
 * Clears the contents of the caches while keeping a holder alive for each device
 */
// The above comment was written by an LLM. 
  void StreamCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // StreamCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

/**
 * Returns the stream cache instance in a thread safe manner 
 */
// The above comment was written by an LLM. 
  StreamCache& getStreamCache() {
    // the public interface is thread safe
    static StreamCache cache;
    return cache;
  }
}  // namespace cms::cuda
