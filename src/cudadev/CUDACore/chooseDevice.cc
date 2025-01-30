#include "chooseDevice.h"
#include "deviceCount.h"

namespace cms::cuda {
/**
 * Returns the device assigned to a given StreamID
 * @param id The EDM stream identifier
 * @return The index of the assigned device
 */
// The above comment was written by an LLM. 
  int chooseDevice(edm::StreamID id) {
    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    return id % deviceCount();
  }
}  // namespace cms::cuda
