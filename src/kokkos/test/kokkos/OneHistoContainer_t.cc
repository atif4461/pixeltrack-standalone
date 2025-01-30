#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "KokkosCore/HistoContainer.h"
using team_policy = Kokkos::TeamPolicy<KokkosExecSpace>;
using member_type = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
/**
 * @brief Main entry point of the program
 *
 * This function initializes random number generators, creates views for device and host memory,
 * and performs various operations such as histogram calculation and window-based searches.
 
void go()
 * @brief Calculate minimum and maximum values for random number generation
 *
 * Depending on the value of NBINS, calculate either the minimum and maximum values of type T
 * or custom limits for the random number distribution.
 
int rmin = std::numeric_limitsT>::min()
int rmax = std::numeric_limitsT>::max()
 * @brief Create random number generator with uniform distribution
 *
 * Initialize a random number engine with a seed value and create a uniform integer distribution
 * within the calculated range [rmin, rmax].
 
std::default_random_engine generator(1234)
std::uniform_int_distributionT> rgen(rmin, rmax)
 * @brief Allocate device and host memory for data storage
 *
 * Create views for device and host memory to store data, where the device view is unmanaged.
 
constexpr uint32_t N = 12000
Kokkos::ViewT*, KokkosExecSpace> v_d_own("v_d", N)
auto v_d = cms::kokkos::make_restrictUnmanaged(v_d_own)
auto v_h = Kokkos::create_mirror_view(v_d_own)
 * @brief Print HistoContainer properties
 *
 * Output the number of bits, bins, capacity, and bin size of the HistoContainer class.
 
std::cout << "HistoContainer " << Hist::nbits() <<'' << Hist::nbins() <<'' << Hist::capacity() <<'' << (rmax - rmin) / Hist::nbins() << std::endl
 * @brief Fill host data array with random numbers
 *
 * Populate the host data array with random numbers generated from the uniform distribution.
 
for (long long j = 0; j  N; j++) v_h(j) = rgen(eng)
 * @brief Copy data from host to device memory
 *
 * Perform a deep copy of the data from the host array to the device array.
 
Kokkos::deep_copy(KokkosExecSpace(), v_d, v_h)
 * @brief Launch parallel kernels for histogram calculation
 *
 * Execute parallel kernels to set zeros, count elements, finalize the histogram, and perform assertions.
 
Kokkos::parallel_for("set_zero",...)
Kokkos::parallel_for("set_zero_bin",...)
Kokkos::parallel_for("count",...)
Kokkos::parallel_for("assert_check",...)
 * @brief Fill histogram with data from device memory
 *
 * Populate the histogram with data from the device array.
 
Kokkos::parallel_for("fill",...)
 * @brief Verify histogram properties
 *
 * Check that the offset at index 0 is zero and the total size of the histogram matches the input data size.
 
assert(0 == histo_h(0).off[0])
assert(N == histo_h(0).size())
 * @brief Validate bin assignments
 *
 * Iterate through the sorted data and verify that bin assignments are correct and non-decreasing.
 
Kokkos::parallel_for("bin",...)
 * @brief Test window-based search functionality
 *
 * Perform window-based searches within the histogram and verify the correctness of the results.
 
Kokkos::parallel_for("forEachInWindow",...)*/
// The above comment was written by an LLM. 
void go() {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::default_random_engine generator(1234);
  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr uint32_t N = 12000;
  Kokkos::View<T*, KokkosExecSpace> v_d_own("v_d", N);
  auto v_d = cms::kokkos::make_restrictUnmanaged(v_d_own);
  auto v_h = Kokkos::create_mirror_view(v_d_own);

  using Hist = cms::kokkos::HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v_h(j) = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v_h(j) = 4;

    Kokkos::deep_copy(KokkosExecSpace(), v_d, v_h);

    printf("start kernel for %d data\n", N);

    using TeamHist = cms::kokkos::HistoContainer<T, NBINS, N, S, uint16_t>;

    Kokkos::View<TeamHist*, KokkosExecSpace> histo_d_own("histo_d", 1);
    auto histo_d = cms::kokkos::make_restrictUnmanaged(histo_d_own);
    auto histo_h = Kokkos::create_mirror_view(histo_d);

    Kokkos::parallel_for(
        "set_zero",
        Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, TeamHist::totbins()),
        KOKKOS_LAMBDA(const int& i) { histo_d(0).off[i] = 0; });

    Kokkos::parallel_for(
        "set_zero_bin",
        Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, TeamHist::capacity()),
        KOKKOS_LAMBDA(const int& i) { histo_d(0).bins[i] = 0; });

    Kokkos::parallel_for(
        "count",
// XXX Team prefix scan used in Hist::finalize() requires power-of-two blockDim
// Kokkos::AUTO() in CUDA backend will somehow use blockDim = 96 thus fail the execution
#if defined KOKKOS_BACKEND_CUDA || defined KOKKOS_BACKEND_HIP
        team_policy(KokkosExecSpace(), 1, 64),
#else
        team_policy(KokkosExecSpace(), 1, Kokkos::AUTO()),
#endif
        KOKKOS_LAMBDA(const member_type& teamMember) {
          for (uint32_t i = teamMember.team_rank(); i < N; i += teamMember.team_size())
            histo_d(0).count(v_d(i));
          teamMember.team_barrier();

          assert(0 == histo_d(0).size());
          teamMember.team_barrier();

          histo_d().finalize(teamMember);
          teamMember.team_barrier();

          assert(N == histo_d(0).size());
        });

    Kokkos::parallel_for(
        "assert_check",
        Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, TeamHist::totbins() - 1),
        KOKKOS_LAMBDA(const int& i) { assert(histo_d(0).off[i] <= histo_d(0).off[i + 1]); });

    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, N), KOKKOS_LAMBDA(const int& i) {
          histo_d(0).fill(v_d(i), i);
        });

    Kokkos::deep_copy(KokkosExecSpace(), histo_h, histo_d);

    assert(0 == histo_h(0).off[0]);
    assert(N == histo_h(0).size());

    Kokkos::parallel_for(
        "bin", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, N - 1), KOKKOS_LAMBDA(const int& i) {
          auto p = histo_d(0).begin() + i;
          assert((*p) < N);
          auto k1 = TeamHist::bin(v_d(*p));
          auto k2 = TeamHist::bin(v_d(*(p + 1)));
          assert(k2 >= k1);
        });

    Kokkos::parallel_for(
        "forEachInWindow", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, N), KOKKOS_LAMBDA(const int& i) {
          auto p = histo_d(0).begin() + i;
          auto j = *p;
          auto b0 = TeamHist::bin(v_d(j));
          int tot = 0;
          auto ftest = [&](int k) {
            assert(k >= 0 && k < static_cast<int>(N));
            ++tot;
          };
          forEachInWindow(histo_d, v_d(j), v_d(j), ftest);
          int rtot = histo_d(0).size(b0);

          assert(tot == rtot);
          tot = 0;
          auto vm = int(v_d(j)) - DELTA;
          auto vp = int(v_d(j)) + DELTA;
          constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
          vm = std::max(vm, 0);
          vm = std::min(vm, vmax);
          vp = std::min(vp, vmax);
          vp = std::max(vp, 0);
          assert(vp >= vm);
          forEachInWindow(histo_d, vm, vp, ftest);
          int bp = TeamHist::bin(vp);
          int bm = TeamHist::bin(vm);
          rtot = histo_d(0).end(bp) - histo_d(0).begin(bm);
          assert(tot == rtot);
        });
  }
}

/**
 * @brief Main program entry point
 * 
 * Initializes Kokkos scope guard and executes test functions with various data types and parameters
 *
 * @return Program exit status
 */
// The above comment was written by an LLM. 
int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  // cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1024*1024*1024);
  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
