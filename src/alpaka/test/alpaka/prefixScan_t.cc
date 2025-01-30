#include <iostream>

#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/initialise.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/prefixScan.h"
#include "AlpakaCore/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

template <typename T>
struct format_traits {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char* failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
struct testPrefixScan {
  template <typename TAcc>
/**
 * Performs parallel operations on an array of elements within a block.
 *
 * @param acc accelerator object
 * @param size number of elements in the block
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, unsigned int size) const {
    auto& ws = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    for_each_element_in_block_strided(acc, size, [&](uint32_t i) { c[i] = 1; });

    alpaka::syncBlockThreads(acc);

    blockPrefixScan(acc, c, co, size, ws);
    blockPrefixScan(acc, c, size, ws);

    assert(1 == c[0]);
    assert(1 == co[0]);

    for_each_element_in_block_strided(acc, size, 1u, [&](uint32_t i) {
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == i + 1);
      assert(c[i] == co[i]);
    });
  }
};

/*
 * NB: GPU-only, so do not care about elements here.
 */
template <typename T>
struct testWarpPrefixScan {
  template <typename TAcc>
/**
 * @brief Performs parallel prefix scan operation on shared variables within a block of threads.
 *
 * @param acc Accelerator handle
 * @param size Size of data being processed
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t size) const {
#if defined(ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND) && defined(__CUDA_ARCH__) || \
    defined(ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND) && defined(__HIP_DEVICE_COMPILE__)
    assert(size <= 32);
    auto& c = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);
    auto& co = alpaka::declareSharedVar<T[1024], __COUNTER__>(acc);

    uint32_t const blockDimension = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    uint32_t const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    auto i = blockThreadIdx;
    c[i] = 1;
    alpaka::syncBlockThreads(acc);
    auto laneId = blockThreadIdx & 0x1f;

    warpPrefixScan(laneId, c, co, i, 0xffffffff);
    warpPrefixScan(laneId, c, i, 0xffffffff);

    alpaka::syncBlockThreads(acc);

    assert(1 == c[0]);
    assert(1 == co[0]);
    if (i != 0) {
      if (c[i] != c[i - 1] + 1)
        printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
      assert(c[i] == c[i - 1] + 1);
      assert(c[i] == static_cast<T>(i + 1));
      assert(c[i] == co[i]);
    }
#endif
  }
};

struct init {
  template <typename TAcc>
/**
 * Performs an operation on a grid of elements setting their values.
 * @param acc access object for the grid
 * @param v pointer to the array of values to be modified
 * @param val value to be assigned to each element in the grid
 * @param n number of elements in the grid
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t* v, uint32_t val, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t index) {
      v[index] = val;

      if (index == 0)
        printf("init\n");
    });
  }
};

struct verify {
  template <typename TAcc>
/**
 * Performs verification of grid elements against an expected sequence.
 *
 * @param[in] acc Accessor object for grid operations
 * @param[in] v Array of values to verify
 * @param[in] n Number of elements in the array
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, uint32_t const* v, uint32_t n) const {
    for_each_element_in_grid(acc, n, [&](uint32_t index) {
      assert(v[index] == index + 1);

      if (index == 0)
        printf("verify\n");
    });
  }
};

/**
 * @brief Main program entry point.
 *
 * This is the primary function where execution begins.
 
  * @brief Initializes the environment.
 
  * @brief Retrieves a reference to the first available device.
 *
 * Returns a constant reference to the device at index 0 from the list of devices for the current platform.
 *
 * @return A constant reference to the device object.
 
  * @brief Creates a new command queue for the specified device.
 *
 * Constructs a new queue object associated with the provided device.
 *
 * @param device The device to create the queue for.
 
  * @brief Configures warp-level parallelism parameters.
 *
 * Defines constants for threads per block and blocks per grid for warp-level operations.
 
  * @brief Calculates the work division for warp-level tasks.
 *
 * Computes the work division based on the number of blocks and threads per block.
 *
 * @tparam Acc1D Accelerator type.
 * @param blocksPerGrid Number of blocks in the grid.
 * @param threadsPerBlockOrElementsPerThread Threads per block or elements per thread.
 * @return Work division object.
 
  * @brief Enqueues a task kernel for warp prefix scan operation.
 *
 * Submits a task kernel to the queue for executing the warp prefix scan operation.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDiv Work division object.
 * @param func Function to execute.
 * @param arg Argument for the function.
 
  * @brief Configures block-level parallelism parameters.
 *
 * Prints a message indicating block-level operations.
 
  * @brief Iterates over different block sizes for portable block prefix scan.
 *
 * Loops through various block sizes for testing purposes.
 
  * @brief Calculates the work division for single-block tasks.
 *
 * Computes the work division based on the number of blocks and threads per block.
 *
 * @tparam Acc1D Accelerator type.
 * @param blocksPerGrid2 Number of blocks in the grid.
 * @param bs Block size or elements per thread.
 * @return Work division object.
 
  * @brief Enqueues a task kernel for portable block prefix scan operation.
 *
 * Submits a task kernel to the queue for executing the portable block prefix scan operation.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDivSingleBlock Work division object.
 * @param func Function to execute.
 * @param arg Argument for the function.
 
  * @brief Tests multiblock prefix scan functionality.
 *
 * Performs multiple iterations of multiblock prefix scan with varying problem sizes.
 
  * @brief Allocates memory buffers on the device.
 *
 * Creates device buffers for storing input and output data.
 *
 * @tparam T Data type.
 * @param queue Command queue.
 * @param numItems Number of items to allocate.
 * @return Device buffer objects.
 
  * @brief Initializes the input data.
 *
 * Enqueues an initialization task kernel to prepare the input data.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDivMultiBlockInit Work division object.
 * @param func Initialization function.
 * @param input_d Input data buffer.
 * @param arg Argument for the function.
 * @param num_items Number of items.
 
  * @brief Performs the first step of multiblock prefix scan.
 *
 * Enqueues a task kernel to perform the initial step of the multiblock prefix scan algorithm.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDivMultiBlock Work division object.
 * @param func First-step function.
 * @param input_d Input data buffer.
 * @param output1_d Output data buffer.
 * @param num_items Number of items.
 
  * @brief Performs the second step of multiblock prefix scan.
 *
 * Enqueues a task kernel to complete the multiblock prefix scan algorithm.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDivMultiBlockSecondStep Work division object.
 * @param func Second-step function.
 * @param input_d Input data buffer.
 * @param output1_d Output data buffer.
 * @param num_items Number of items.
 * @param nBlocks Number of blocks.
 
  * @brief Verifies the results.
 *
 * Enqueues a verification task kernel to validate the output.
 *
 * @tparam Acc1D Accelerator type.
 * @param workDivMultiBlock Work division object.
 * @param func Verification function.
 * @param output1_d Output data buffer.
 * @param num_items Number of items.
 */
// The above comment was written by an LLM. 
int main() {
  initialise();
  Device const& device = devices<Platform>().at(0);
  Queue queue(device);

  // WARP PREFIXSCAN (OBVIOUSLY GPU-ONLY)
#if defined(ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND) || defined(ALPAKA_ACC_GPU_HIP_ASYNC_BACKEND)
  std::cout << "warp level" << std::endl;

  const auto threadsPerBlockOrElementsPerThread = 32;
  const auto blocksPerGrid = 1;
  const auto workDivWarp = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 32));
  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 16));
  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivWarp, testWarpPrefixScan<int>(), 5));
#endif

  // PORTABLE BLOCK PREFIXSCAN
  std::cout << "block level" << std::endl;

  // Running kernel with 1 block, and bs threads per block or elements per thread.
  // NB: obviously for tests only, for perf would need to use bs = 1024 in GPU version.
  for (int bs = 32; bs <= 1024; bs += 32) {
    const auto blocksPerGrid2 = 1;
    const auto workDivSingleBlock = make_workdiv<Acc1D>(blocksPerGrid2, bs);

    std::cout << "blocks per grid: " << blocksPerGrid2 << ", threads per block or elements per thread: " << bs
              << std::endl;

    // Problem size
    for (int j = 1; j <= 1024; ++j) {
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<uint16_t>(), j));
      alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivSingleBlock, testPrefixScan<float>(), j));
    }
  }

  // PORTABLE MULTI-BLOCK PREFIXSCAN
  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    std::cout << "multiblock" << std::endl;
    num_items *= 10;

    auto input_d = make_device_buffer<uint32_t[]>(queue, num_items);
    auto output1_d = make_device_buffer<uint32_t[]>(queue, num_items);

    const auto nThreadsInit = 256;  // NB: 1024 would be better
    const auto nBlocksInit = divide_up_by(num_items, nThreadsInit);
    const auto workDivMultiBlockInit = make_workdiv<Acc1D>(nBlocksInit, nThreadsInit);

    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDivMultiBlockInit, init(), input_d.data(), 1, num_items));

    const auto nThreads = 1024;
    const auto nBlocks = divide_up_by(num_items, nThreads);
    const auto workDivMultiBlock = make_workdiv<Acc1D>(nBlocks, nThreads);

    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nBlocks << std::endl;
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            workDivMultiBlock, multiBlockPrefixScanFirstStep<uint32_t>(), input_d.data(), output1_d.data(), num_items));

    const auto blocksPerGridSecondStep = 1;
    const auto workDivMultiBlockSecondStep = make_workdiv<Acc1D>(blocksPerGridSecondStep, nThreads);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDivMultiBlockSecondStep,
                                                    multiBlockPrefixScanSecondStep<uint32_t>(),
                                                    input_d.data(),
                                                    output1_d.data(),
                                                    num_items,
                                                    nBlocks));

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivMultiBlock, verify(), output1_d.data(), num_items));

    alpaka::wait(queue);  // input_d and output1_d end of scope
  }                       // ksize

  return 0;
}
