#include <iostream>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

/**
 * @brief Main program entry point
 *
 * Initializes the Kokkos scope guard and performs a parallel for loop operation
 * using the KokkosExecSpace backend
 *
 * @return Integer value indicating successful execution
 
 * @brief Parallel for loop iteration callback
 *
 * Prints the current loop index during parallel execution
 *
 * @param i Current loop index
 */
// The above comment was written by an LLM. 
int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  std::cout << "World" << std::endl;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<KokkosExecSpace>(0, 4),
      KOKKOS_LAMBDA(const size_t i) { printf("Kokkos::parallel_for loop element %lu\n", i); });
  return 0;
}
