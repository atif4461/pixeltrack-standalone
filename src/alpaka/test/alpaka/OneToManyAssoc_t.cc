#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "AlpakaCore/HistoContainer.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/initialise.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/workdivision.h"

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using Assoc = OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = OneToManyAssoc<uint16_t, 8, MaxTk>;
using TK = std::array<uint16_t, 4>;

struct countMultiLocal {
  template <typename TAcc>
/**
 * @brief Performs operation on grid elements with strided access pattern
 * @param acc accelerator handle
 * @param tk kernel data
 * @param assoc multiplicity association
 * @param n number of elements in grid
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t i) {
      auto& local = alpaka::declareSharedVar<Multiplicity::CountersOnly, __COUNTER__>(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const bool oncePerSharedMemoryAccess = (threadIdxLocal == 0);
      if (oncePerSharedMemoryAccess) {
        local.zero();
      }
      alpaka::syncBlockThreads(acc);
      local.countDirect(acc, 2 + i % 4);
      alpaka::syncBlockThreads(acc);
      if (oncePerSharedMemoryAccess) {
        assoc->add(acc, local);
      }
    });
  }
};

struct countMulti {
  template <typename TAcc>
/**
 * @brief Performs an operation on a grid with strided access pattern
 * @param acc Accessor object for the grid
 * @param tk Pointer to kernel data
 * @param assoc Object for handling multiplicity associations
 * @param n Number of elements in the grid
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Multiplicity* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t i) { assoc->countDirect(acc, 2 + i % 4); });
  }
};

struct verifyMulti {
  template <typename TAcc>
/**
 * Checks if two multiplicity objects have the same offset values for all bins in the grid.
 *
 * @param acc Accessor object for parallel execution
 * @param m1 First multiplicity object to compare
 * @param m2 Second multiplicity object to compare
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) const {
    for_each_element_in_grid_strided(
        acc, Multiplicity::totbins(), [&](uint32_t i) { assert(m1->off[i] == m2->off[i]); });
  }
};

struct count {
  template <typename TAcc>
/**
 * @brief Iterates over grid elements with stride and updates association counts
 * @param acc access object for iteration
 * @param tk input data array
 * @param assoc association object for updating counts
 * @param n number of iterations
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->countDirect(acc, tk[k][j]);
      }
    });
  }
};

struct fill {
  template <typename TAcc>
/**
 * Brief description of the function 
 * Detailed description of the function that takes an accelerator 
 * and kernel data and performs operation on association object
 *
 * @param[in] acc Accelerator object used for computation
 * @param[in] tk Kernel data used in computation
 * @param[out] assoc Association object being updated
 * @param[in] n Size parameter for kernel data
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                TK const* __restrict__ tk,
                                Assoc* __restrict__ assoc,
                                uint32_t n) const {
    for_each_element_in_grid_strided(acc, 4 * n, [&](uint32_t i) {
      auto k = i / 4;
      auto j = i - 4 * k;
      assert(j < 4);
      if (k >= n) {
        return;
      }
      if (tk[k][j] < MaxElem) {
        assoc->fillDirect(acc, tk[k][j], k);
      }
    });
  }
};

struct verify {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc* __restrict__ assoc) const {
    assert(assoc->size() < Assoc::capacity());
  }
};

struct fillBulk {
  template <typename TAcc, typename Assoc>
/**
 * Performs operation on grid elements with striding
 * @param acc accelerator object
 * @param apc atomic pair counter pointer
 * @param tk input data array
 * @param assoc association object
 * @param n number of elements in grid
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(
      const TAcc& acc, AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, uint32_t n) const {
    for_each_element_in_grid_strided(acc, n, [&](uint32_t k) {
      auto m = tk[k][3] < MaxElem ? 4 : 3;
      assoc->bulkFill(acc, *apc, &tk[k][0], m);
    });
  }
};

struct verifyBulk {
  template <typename TAcc, typename Assoc>
/**
 * Checks for overflow condition in atomic pair counter and validates associative array size.
 */
// The above comment was written by an LLM. 
  ALPAKA_FN_ACC void operator()(const TAcc& acc, Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) const {
    if (apc->get().m >= Assoc::nbins()) {
      printf("Overflow %d %d\n", apc->get().m, Assoc::nbins());
    }
    assert(assoc->size() < Assoc::capacity());
  }
};

/**

### Function Comments
 * @brief The main entry point of the program.
 *
 * Initializes the environment, sets up the device and queue, generates random data,
 * performs various operations on the data, and verifies the results.
 *
 * @return An integer indicating the program's exit status.
 
 
  * @brief Launches the zero kernel on the specified device data.
 *
 * Initializes the device data to zero.
 *
 * @param data The device data to be initialized.
 * @param queue The command queue used to execute the kernel.
 
 
  * @brief Fills the bulk data structure on the device.
 *
 * Copies data from one buffer to another on the device.
 *
 * @param workDiv The work division for the kernel launch.
 * @param data The destination buffer.
 * @param src The source buffer.
 * @param N The number of elements in the buffers.
 
 
  * @brief Finalizes the device data after all operations have been completed.
 *
 * Ensures that all operations on the device data have finished before proceeding.
 *
 * @param data The device data to be finalized.
 * @param queue The command queue used to execute the kernel.
 
 
  * @brief Verifies the contents of the device data.
 *
 * Checks the device data against expected values.
 *
 * @param data The device data to be verified.
 
 
  * @brief Counts the multiplicity of elements in the data.
 *
 * Determines the frequency of occurrence of each element in the data.
 *
 * @param workDiv The work division for the kernel launch.
 * @param data The input data.
 * @param dest The destination buffer for the counts.
 * @param N The number of elements in the data.
 
 
  * @brief Counts the multiplicity of elements in the data using local counters.
 *
 * Determines the frequency of occurrence of each element in the data using block-local counters.
 *
 * @param workDiv The work division for the kernel launch.
 * @param data The input data.
 * @param dest The destination buffer for the counts.
 * @param N The number of elements in the data.
 
 
  * @brief Verifies the multiplicity counts.
 *
 * Compares the counts obtained using global and local counters.
 *
 * @param m1 The first set of counts.
 * @param m2 The second set of counts.
 */
// The above comment was written by an LLM. 
int main() {
  initialise();
  const Device device = devices<Platform>().at(0);
  Queue queue(device);

  std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::nbins() << ' ' << Assoc::capacity() << std::endl;
  std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::nbins() << ' '
            << SmallAssoc::capacity() << std::endl;

  std::mt19937 eng;
  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  auto tr = make_host_buffer<std::array<uint16_t, 4>[]>(queue, N);
  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

  auto v_d = make_device_buffer<std::array<uint16_t, 4>[]>(queue, N);
  alpaka::memcpy(queue, v_d, tr);

  auto a_d = make_device_buffer<Assoc>(queue);
  alpaka::memset(queue, a_d, 0);

  const auto threadsPerBlockOrElementsPerThread = 256u;
  const auto blocksPerGrid4N = divide_up_by(4 * N, threadsPerBlockOrElementsPerThread);
  const auto workDiv4N = make_workdiv<Acc1D>(blocksPerGrid4N, threadsPerBlockOrElementsPerThread);

  launchZero<Acc1D>(a_d.data(), queue);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, count(), v_d.data(), a_d.data(), N));

  launchFinalize<Acc1D>(a_d.data(), queue);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verify(), a_d.data()));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, fill(), v_d.data(), a_d.data(), N));

  auto la = make_host_buffer<Assoc>(queue);
  alpaka::memcpy(queue, la, a_d);
  alpaka::wait(queue);

  std::cout << la->size() << std::endl;
  imax = 0;
  ave = 0;
  z = 0;
  for (auto i = 0U; i < n; ++i) {
    auto x = la->size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(n));
  std::cout << "found with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  auto dc_d = make_device_buffer<AtomicPairCounter>(queue);
  alpaka::memset(queue, dc_d, 0);

  const auto blocksPerGrid = divide_up_by(N, threadsPerBlockOrElementsPerThread);
  const auto workDiv = make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), a_d.data(), N));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, finalizeBulk(), dc_d.data(), a_d.data()));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), a_d.data(), dc_d.data()));

  alpaka::memcpy(queue, la, a_d);

  auto dc = make_host_buffer<AtomicPairCounter>(queue);
  alpaka::memcpy(queue, dc, dc_d);
  alpaka::wait(queue);

  alpaka::memset(queue, dc_d, 0);
  auto sa_d = make_device_buffer<SmallAssoc>(queue);
  alpaka::memset(queue, sa_d, 0);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, fillBulk(), dc_d.data(), v_d.data(), sa_d.data(), N));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, finalizeBulk(), dc_d.data(), sa_d.data()));

  alpaka::enqueue(queue,
                  alpaka::createTaskKernel<Acc1D>(WorkDiv1D{1u, 1u, 1u}, verifyBulk(), sa_d.data(), dc_d.data()));

  std::cout << "final counter value " << dc->get().n << ' ' << dc->get().m << std::endl;

  std::cout << la->size() << std::endl;
  imax = 0;
  ave = 0;
  for (auto i = 0U; i < N; ++i) {
    auto x = la->size(i);
    if (!(x == 4 || x == 3)) {
      std::cout << i << ' ' << x << std::endl;
    }
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(N));
  std::cout << "found with ave occupancy " << double(ave) / N << ' ' << imax << std::endl;

  // here verify use of block local counters
  auto m1_d = make_device_buffer<Multiplicity>(queue);
  alpaka::memset(queue, m1_d, 0);
  auto m2_d = make_device_buffer<Multiplicity>(queue);
  alpaka::memset(queue, m2_d, 0);

  launchZero<Acc1D>(m1_d.data(), queue);
  launchZero<Acc1D>(m2_d.data(), queue);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMulti(), v_d.data(), m1_d.data(), N));

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv4N, countMultiLocal(), v_d.data(), m2_d.data(), N));

  const auto blocksPerGridTotBins = 1u;
  const auto threadsPerBlockOrElementsPerThreadTotBins = Multiplicity::totbins();
  const auto workDivTotBins = make_workdiv<Acc1D>(blocksPerGridTotBins, threadsPerBlockOrElementsPerThreadTotBins);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

  launchFinalize<Acc1D>(m1_d.data(), queue);
  launchFinalize<Acc1D>(m2_d.data(), queue);

  alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDivTotBins, verifyMulti(), m1_d.data(), m2_d.data()));

  alpaka::wait(queue);

  return 0;
}
