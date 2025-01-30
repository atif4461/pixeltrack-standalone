#include "CUDACore/EventCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/eventWorkHasCompleted.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::hip {
/**
 * Destroys a hip event object and releases associated resources.
 */
// The above comment was written by an LLM. 
  void EventCache::Deleter::operator()(hipEvent_t event) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      cudaCheck(hipEventDestroy(event));
    }
  }

  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(deviceCount()) {}

/**
 * Retrieves an event from the cache, ensuring that the captured work has been completed.
 *
 * @return A SharedEventPtr representing the retrieved event.
 */
// The above comment was written by an LLM. 
  SharedEventPtr EventCache::get() {
    const auto dev = currentDevice();
    auto event = makeOrGet(dev);
    // captured work has completed, or a just-created event
    if (eventWorkHasCompleted(event.get())) {
      return event;
    }

    // Got an event with incomplete captured work. Try again until we
    // get a completed (or a just-created) event. Need to keep all
    // incomplete events until a completed event is found in order to
    // avoid ping-pong with an incomplete event.
    std::vector<SharedEventPtr> ptrs{std::move(event)};
    bool completed;
    do {
      event = makeOrGet(dev);
      completed = eventWorkHasCompleted(event.get());
      if (not completed) {
        ptrs.emplace_back(std::move(event));
      }
    } while (not completed);
    return event;
  }

/**
 * Returns a shared pointer to an event object for the specified device 
 * creating one if it does not already exist in the cache
 * @param dev device identifier
 * @return SharedEventPtr to the event object
 */
// The above comment was written by an LLM. 
  SharedEventPtr EventCache::makeOrGet(int dev) {
    return cache_[dev].makeOrGet([dev]() {
      hipEvent_t event;
      // it should be a bit faster to ignore timings
      cudaCheck(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      return std::unique_ptr<BareEvent, Deleter>(event, Deleter{dev});
    });
  }

/**
 * Clears the contents of the event cache while preserving device holders
 */
// The above comment was written by an LLM. 
  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

/**
 * Returns the event cache instance in a thread safe manner 
 */
// The above comment was written by an LLM. 
  EventCache& getEventCache() {
    // the public interface is thread safe
    static EventCache cache;
    return cache;
  }
}  // namespace cms::hip
