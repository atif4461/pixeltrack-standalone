#include <iostream>

#include "KokkosCore/kokkosConfig.h"

/**
 * @brief Main program entry point
 * @return Integer indicating program execution status
 */
// The above comment was written by an LLM. 
int main() {
  std::cout << "Hello from "
#ifdef KOKKOS_BACKEND_SERIAL
            << "CPU serial"
#elif defined KOKKOS_BACKEND_CUDA
            << "CUDA"
#endif
            << " backend" << std::endl;
  return 0;
}
