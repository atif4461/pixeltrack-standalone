#include <cassert>
#include <iostream>

#include "AlpakaCore/AtomicPairCounter.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/initialise.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct update {
  template <typename TAcc>
/**
 * @brief Performs an operation on a grid with atomic pair counter and updates indices and container.
 * @param acc Accessor object for the grid.
 * @param dc Pointer to an AtomicPairCounter object.
 * @param ind Array of indices to be updated.
 * @param cont Array of containers to be updated.
 * @param n Size of the grid.
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t i) {
      auto m = i % 11;
      m = m % 6 + 1;  // max 6, no 0
      auto c = dc->add(acc, m);
      assert(c.m < n);
      ind[c.m] = c.n;
      for (uint32_t j = c.n; j < c.n + m; ++j)
        cont[j] = i;
    });
  }
};

struct finalize {
  template <typename TAcc>
/**
 * Brief description of the function 
 * @param acc access object
 * @param dc pointer to atomic pair counter
 * @param ind array of indices
 * @param cont array of counters
 * @param n size parameter
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) const {
    assert(dc->get().m == n);
    ind[n] = dc->get().n;
  }
};

struct verify {
  template <typename TAcc>
/**
 * @brief Checks the consistency of index and container arrays within a grid.
 *
 * @param acc access object for parallel execution
 * @param dc pointer to atomic pair counter
 * @param ind array of indices
 * @param cont array of container values
 * @param n size of the grid
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(
      const TAcc &acc, AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t i) {
      assert(0 == ind[0]);
      assert(dc->get().m == n);
      assert(ind[n] == dc->get().n);
      auto ib = ind[i];
      auto ie = ind[i + 1];
#ifndef NDEBUG
      auto k = cont[ib++];
#endif
      assert(k < n);
      for (; ib < ie; ++ib)
        assert(cont[ib] == k);
    });
  }
};

/**
 * @brief Main program entry point
 *
 * This is the primary function where program execution begins
   * @brief Initializes the application environment
  * @brief Returns a list of available devices for the current platform
 *
 * @return A collection of devices compatible with the current platform
  * @brief Creates a new command queue object
 *
 * @param device The device associated with this queue
  * @brief Allocates memory on the device for an AtomicPairCounter object
 *
 * @param queue The command queue used for allocation
 * @return A buffer containing the allocated AtomicPairCounter object
  * @brief Sets all bytes in the specified device memory to zero
 *
 * @param queue The command queue used for the operation
 * @param memPtr Pointer to the device memory to be cleared
  * @brief Prints the size of the AtomicPairCounter type to the console
  * @brief Defines constants representing array sizes
  * @brief Allocates memory on the device for arrays of uint32_t values
 *
 * @param queue The command queue used for allocation
 * @param size The number of elements in the array
 * @return A buffer containing the allocated array
  * @brief Specifies the number of values processed by the kernel functions
  * @brief Configures the block and thread counts for kernel launch
  * @brief Calculates the work division for kernel execution
 *
 * @param blocksPerGrid Number of blocks in the grid
 * @param threadsPerBlockOrElementsPerThread Number of threads per block or elements per thread
 * @return The calculated work division
  * @brief Enqueues a task kernel for execution on the device
 *
 * @param queue The command queue used for enqueueing
 * @param kernel The kernel to be executed
 * @param args Kernel arguments
  * @brief Updates data on the device using the specified kernel function
 *
 * @param c_d Counter data
 * @param n_d Array of N uint32_t values
 * @param m_d Array of M uint32_t values
 * @param numValues Number of values processed
  * @brief Finalizes data on the device using the specified kernel function
 *
 * @param c_d Counter data
 * @param n_d Array of N uint32_t values
 * @param m_d Array of M uint32_t values
 * @param numValues Number of values processed
  * @brief Verifies data on the device using the specified kernel function
 *
 * @param c_d Counter data
 * @param n_d Array of N uint32_t values
 * @param m_d Array of M uint32_t values
 * @param numValues Number of values processed
  * @brief Allocates host memory for an AtomicPairCounter object
 *
 * @param queue The command queue used for allocation
 * @return A buffer containing the allocated AtomicPairCounter object
  * @brief Copies data from device memory to host memory
 *
 * @param queue The command queue used for the transfer
 * @param dst Host memory destination
 * @param src Device memory source
  * @brief Waits for completion of all commands enqueued on the device
 *
 * @param queue The command queue to wait on
  * @brief Prints counter values to the console
 */
// The above comment was written by an LLM. 
int main() {
  initialise();
  const Device device = devices<Platform>().at(0);
  Queue queue(device);

  auto c_d = make_device_buffer<AtomicPairCounter>(queue);
  alpaka::memset(queue, c_d, 0);

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  auto n_d = make_device_buffer<uint32_t[]>(queue, N);
  auto m_d = make_device_buffer<uint32_t[]>(queue, M);

  constexpr uint32_t NUM_VALUES = 10000;

  // Update
  const auto blocksPerGrid = 2000u;
  const auto threadsPerBlockOrElementsPerThread = 512u;
  const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv, update(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  // Finalize
  const auto blocksPerGridFinalize = 1u;
  const auto threadsPerBlockOrElementsPerThreadFinalize = 1u;
  const auto workDivFinalize = make_workdiv<Acc1D>(blocksPerGridFinalize, threadsPerBlockOrElementsPerThreadFinalize);
  alpaka::enqueue(
      queue,
      alpaka::createTaskKernel<Acc1D>(workDivFinalize, finalize(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  // Verify
  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(workDiv, verify(), c_d.data(), n_d.data(), m_d.data(), NUM_VALUES));

  auto c_h = make_host_buffer<AtomicPairCounter>(queue);
  alpaka::memcpy(queue, c_h, c_d);
  alpaka::wait(queue);

  std::cout << c_h->get().n << ' ' << c_h->get().m << std::endl;

  return 0;
}
