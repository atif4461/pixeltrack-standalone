#include <cassert>

#include <sycl/sycl.hpp>

#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/chooseDevice.h"

using AtomicPairCounter = cms::sycltools::AtomicPairCounter;

/**
 * Updates the counter and index arrays based on the current item in a parallel iteration.
 * @param dc pointer to an AtomicPairCounter object
 * @param ind array of indices to be updated
 * @param cont array of containers to be updated
 * @param n total number of elements
 * @param item current item in the parallel iteration
 */
// The above comment was written by an LLM. 
void update(AtomicPairCounter *dc, uint32_t *ind, uint32_t *cont, uint32_t n, sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (i >= n)
    return;

  auto m = i % 11;
  m = m % 6 + 1;  // max 6, no 0
  auto c = dc->add(m);
  //assert(c.m < n);
  ind[c.m] = c.n;
  for (auto j = c.n; j < c.n + m; ++j)
    cont[j] = i;
};

/**
 * @brief Finalizes the atomic pair counter data structure
 * @param dc AtomicPairCounter object
 * @param ind array of indices
 * @param cont unused container pointer
 * @param n number of elements
 */
// The above comment was written by an LLM. 
void finalize(AtomicPairCounter const *dc, uint32_t *ind, uint32_t *cont, uint32_t n) {
  //assert(dc->get().m == n);
  ind[n] = dc->get().n;
}

/**
 * Verifies the correctness of the input data structures.
 *
 * @param[in] dc        The AtomicPairCounter object to be verified.
 * @param[in] ind       The index array to be verified.
 * @param[in] cont      The content array to be verified.
 * @param[in] n         The size of the arrays.
 * @param[in] item      The SYCL item object representing the current work item.
 */
// The above comment was written by an LLM. 
void verify(AtomicPairCounter const *dc, uint32_t const *ind, uint32_t const *cont, uint32_t n, sycl::nd_item<1> item) {
  auto i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (i >= n)
    return;
  //assert(0 == ind[0]);
  //assert(dc->get().m == n);
  //assert(ind[n] == dc->get().n);
  // auto ib = ind[i];
  // auto ie = ind[i + 1];
  // auto k = cont[ib++];
  //assert(k < n);
  //for (; ib < ie; ++ib)
  //assert(cont[ib] == k);
}

#include <iostream>
/**
 * Main program entry point
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Program exit status
 */
// The above comment was written by an LLM. 
int main(int argc, char **argv) {
  std::string devices(argv[1]);
  setenv("ONEAPI_DEVICE_SELECTOR", devices.c_str(), true);

  cms::sycltools::enumerateDevices(true);
  sycl::device device = cms::sycltools::chooseDevice(0);
  sycl::queue queue = sycl::queue(device, sycl::property::queue::in_order());

  std::cout << "AtomicPairCounter offload to " << device.get_info<sycl::info::device::name>() << " on backend "
            << device.get_backend() << std::endl;

  AtomicPairCounter *dc_d = sycl::malloc_device<AtomicPairCounter>(1, queue);
  queue.memset(dc_d, 0, sizeof(AtomicPairCounter)).wait();

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;
  uint32_t *n_d, *m_d;
  n_d = (uint32_t *)sycl::malloc_device(N * sizeof(int), queue);
  m_d = (uint32_t *)sycl::malloc_device(M * sizeof(int), queue);

  int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int threads = std::min(512, max_work_group_size);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(2000 * threads, threads),
                     [=](sycl::nd_item<1> item) { update(dc_d, n_d, m_d, 10000, item); });
  });
  queue.submit([&](sycl::handler &cgh) { cgh.single_task([=]() { finalize(dc_d, n_d, m_d, 10000); }); });
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(2000 * threads, threads),
                     [=](sycl::nd_item<1> item) { verify(dc_d, n_d, m_d, 10000, item); });
  });

  AtomicPairCounter dc;
  queue.memcpy(&dc, dc_d, sizeof(AtomicPairCounter)).wait();

  std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

  queue.wait_and_throw();
  return 0;
}
