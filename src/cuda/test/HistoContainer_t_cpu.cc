#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/requireDevices.h"

using namespace cms::cuda;

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
/**
 * @brief Function to test histogram functionality
 *
 * This function tests various aspects of histogram creation, 
 * data insertion, and retrieval. It uses random number generation 
 * to populate the histograms and verifies their correctness.
  
 * @brief Lambda function to verify histogram bin contents
 *
 * This lambda function is used to verify that the elements in a 
 * particular range of bins satisfy certain conditions.
 *
 * @param i The current bin index
 * @param j The neighboring bin index
 * @param k The index of the element being verified
 * @param t1 The first index to compare with
 * @param t2 The second index to compare with
 
 
 * @brief Function to iterate over all elements in specified bins
 *
 * This function iterates over all elements within a specified 
 * distance from a given bin and applies a callback function.
 *
 * @param hist The histogram object
 * @param val The value around which to search
 * @param width The distance from the bin to consider
 * @param func The callback function to apply
 */
// The above comment was written by an LLM. 
void go() {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;
  T v[N];

  using Hist = HistoContainer<T, NBINS, N, S>;
  using Hist4 = HistoContainer<T, NBINS, N, S, uint16_t, 4>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist::capacity() << ' ' << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;
  std::cout << "HistoContainer4 " << Hist4::nbits() << ' ' << Hist4::nbins() << ' ' << Hist4::totbins() << ' '
            << Hist4::capacity() << ' ' << (rmax - rmin) / Hist::nbins() << std::endl;
  for (auto nh = 0; nh < 4; ++nh)
    std::cout << "bins " << int(Hist4::bin(0)) + Hist4::histOff(nh) << ' ' << int(Hist::bin(rmin)) + Hist4::histOff(nh)
              << ' ' << int(Hist::bin(rmax)) + Hist4::histOff(nh) << std::endl;

  Hist h;
  Hist4 h4;
  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;
    h.zero();
    h4.zero();
    assert(h.size() == 0);
    assert(h4.size() == 0);
    for (long long j = 0; j < N; j++) {
      h.count(v[j]);
      if (j < 2000)
        h4.count(v[j], 2);
      else
        h4.count(v[j], j % 4);
    }
    assert(h.size() == 0);
    assert(h4.size() == 0);
    h.finalize();
    h4.finalize();
    assert(h.size() == N);
    assert(h4.size() == N);
    for (long long j = 0; j < N; j++) {
      h.fill(v[j], j);
      if (j < 2000)
        h4.fill(v[j], j, 2);
      else
        h4.fill(v[j], j, j % 4);
    }
    assert(h.off[0] == 0);
    assert(h4.off[0] == 0);
    assert(h.size() == N);
    assert(h4.size() == N);

    auto verify = [&](uint32_t i, uint32_t j, uint32_t k, uint32_t t1, uint32_t t2) {
      assert((int32_t)t1 < N);
      assert((int32_t)t2 < N);
      if (i != j && T(v[t1] - v[t2]) <= 0)
        std::cout << "for " << i << ':' << v[k] << " failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    for (uint32_t i = 0; i < Hist::nbins(); ++i) {
      if (0 == h.size(i))
        continue;
      auto k = *h.begin(i);
      assert(k < N);
      auto kl = NBINS != 128 ? h.bin(std::max(rmin, v[k] - DELTA)) : h.bin(v[k] - T(DELTA));
      auto kh = NBINS != 128 ? h.bin(std::min(rmax, v[k] + DELTA)) : h.bin(v[k] + T(DELTA));
      if (NBINS == 128) {
        assert(kl != i);
        assert(kh != i);
      }
      if (NBINS != 128) {
        assert(kl <= i);
        assert(kh >= i);
      }
      // std::cout << kl << ' ' << kh << std::endl;
      for (auto j = h.begin(kl); j < h.end(kl); ++j)
        verify(i, kl, k, k, (*j));
      for (auto j = h.begin(kh); j < h.end(kh); ++j)
        verify(i, kh, k, (*j), k);
    }
  }

  for (long long j = 0; j < N; j++) {
    auto b0 = h.bin(v[j]);
    int w = 0;
    int tot = 0;
    auto ftest = [&](int k) {
      assert(k >= 0 && k < N);
      tot++;
    };
    forEachInBins(h, v[j], w, ftest);
    int rtot = h.end(b0) - h.begin(b0);
    assert(tot == rtot);
    w = 1;
    tot = 0;
    forEachInBins(h, v[j], w, ftest);
    int bp = b0 + 1;
    int bm = b0 - 1;
    if (bp < int(h.nbins()))
      rtot += h.end(bp) - h.begin(bp);
    if (bm >= 0)
      rtot += h.end(bm) - h.begin(bm);
    assert(tot == rtot);
    w = 2;
    tot = 0;
    forEachInBins(h, v[j], w, ftest);
    bp++;
    bm--;
    if (bp < int(h.nbins()))
      rtot += h.end(bp) - h.begin(bp);
    if (bm >= 0)
      rtot += h.end(bm) - h.begin(bm);
    assert(tot == rtot);
  }
}

/**
 * @brief Main program entry point
 *
 * @return Integer value indicating program execution status
  
 * @brief Template function to perform operations on different data types
 *
 * @tparam T Data type to be used in the operation
 *
 * @return None
 
 * @brief Specialization of template function for specific data type with additional parameters
 *
 * @tparam T Data type to be used in the operation
 * @tparam U First parameter value
 * @tparam V Second parameter value
 * @tparam W Third parameter value
 *
 * @return None
 */
// The above comment was written by an LLM. 
int main() {
  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}
