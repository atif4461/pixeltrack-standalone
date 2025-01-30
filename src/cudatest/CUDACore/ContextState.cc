#include "CUDACore/ContextState.h"

#include <stdexcept>

namespace cms::cuda {
/**
 * Throws an exception if the object is currently holding a valid stream state
 */
// The above comment was written by an LLM. 
  void ContextState::throwIfStream() const {
    if (stream_) {
      throw std::runtime_error("Trying to set ContextState, but it already had a valid state");
    }
  }

/**
 * Throws an exception if the object is not associated with a valid stream
 */
// The above comment was written by an LLM. 
  void ContextState::throwIfNoStream() const {
    if (not stream_) {
      throw std::runtime_error("Trying to get ContextState, but it did not have a valid state");
    }
  }
}  // namespace cms::cuda
