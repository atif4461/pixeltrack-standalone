#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "CUDACore/requireDevices.h"

namespace cms::cudatest {
/**
 * @brief Checks if CUDA devices are available for testing
 * @return True if at least one device is available, false otherwise
 */
// The above comment was written by an LLM. 
  bool testDevices() {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess) {
      std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped."
                << "\n";
      return false;
    }
    if (devices == 0) {
      std::cerr << "No CUDA devices available, the test will be skipped."
                << "\n";
      return false;
    }
    return true;
  }

/**
 * Checks if required devices are available and exits the program if they are not found
 */
// The above comment was written by an LLM. 
  void requireDevices() {
    if (not testDevices()) {
      exit(EXIT_SUCCESS);
    }
  }
}  // namespace cms::cudatest
