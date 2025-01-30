#include "CUDACore/ContextState.h"

#include <stdexcept>

namespace cms::hip {
/**
 * Checks if the object has an existing stream and throws an exception if it does. 
 */
// The above comment was written by an LLM. 
  void ContextState::throwIfStream() const {
    if (stream_) {
      throw std::runtime_error("Trying to set ContextState, but it already had a valid state");
    }
  }

/**
 * Throws an exception if the object does not have a valid stream associated with it
 */
// The above comment was written by an LLM. 
  void ContextState::throwIfNoStream() const {
    if (not stream_) {
      throw std::runtime_error("Trying to get ContextState, but it did not have a valid state");
    }
  }
}  // namespace cms::hip
