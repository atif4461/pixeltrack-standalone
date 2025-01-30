#include "kokkosAlgo1.h"

namespace {
  constexpr int NUM_VALUES = 1000;

/**
 * @brief Computes the product of two vectors at specific positions.
 * @param[in] a First input vector.
 * @param[in] b Second input vector.
 * @param[out] c Resulting value from vector product.
 * @param[in] row Row index for first vector.
 * @param[in] col Column index for second vector.
 */
// The above comment was written by an LLM. 
  KOKKOS_INLINE_FUNCTION void vectorProd(const Kokkos::View<float*, KokkosExecSpace> a,
                                         const Kokkos::View<float*, KokkosExecSpace> b,
                                         Kokkos::View<float**, KokkosExecSpace> c,
                                         size_t row,
                                         size_t col) {
    c(row, col) = a[row] * b[col];
  }
}  // namespace

namespace KOKKOS_NAMESPACE {
/**
 * @brief Initializes device views with host data and performs parallel operations
 * @param execSpace Execution space for Kokkos operations
 * @return Device view of floats
 
 
  * @brief Performs element wise addition of two vectors
 * @param[in] d_a First vector
 * @param[in] d_b Second vector
 * @param[out] d_c Resultant vector
 * @param[in] i Index for parallel iteration
  
  * @brief Computes outer product of two vectors
 * @param[in] d_a First vector
 * @param[in] d_b Second vector
 * @param[out] d_ma Resultant matrix
 * @param[in] row Row index for parallel iteration
 * @param[in] col Column index for parallel iteration
 
  * @brief Computes outer product of two vectors
 * @param[in] d_a First vector
 * @param[in] d_c Second vector
 * @param[out] d_mb Resultant matrix
 * @param[in] row Row index for parallel iteration
 * @param[in] col Column index for parallel iteration
 
  * @brief Computes matrix product of two matrices
 * @param[in] d_ma First matrix
 * @param[in] d_mb Second matrix
 * @param[out] d_mc Resultant matrix
 * @param[in] row Row index for parallel iteration
 * @param[in] col Column index for parallel iteration
 
  * @brief Computes dot product of a matrix and a vector
 * @param[in] d_ma Matrix
 * @param[in] d_b Vector
 * @param[out] d_c Resultant vector
 * @param[in] row Row index for parallel iteration
 */
// The above comment was written by an LLM. 
  Kokkos::View<float*, KokkosExecSpace> kokkosAlgo1(KokkosExecSpace const& execSpace) {
    Kokkos::View<float*, KokkosExecSpace> d_a{"d_a", NUM_VALUES};
    Kokkos::View<float*, KokkosExecSpace> d_b{"d_b", NUM_VALUES};

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (int i = 0; i < NUM_VALUES; i++) {
      h_a[i] = i;
      h_b[i] = i * i;
    }

    Kokkos::deep_copy(execSpace, d_a, h_a);
    Kokkos::deep_copy(execSpace, d_b, h_b);

    Kokkos::View<float*, KokkosExecSpace> d_c{"d_c", NUM_VALUES};
    Kokkos::parallel_for(
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, NUM_VALUES)),
        KOKKOS_LAMBDA(const size_t i) { d_c[i] = d_a[i] + d_b[i]; });

    Kokkos::View<float**, KokkosExecSpace> d_ma{"d_ma", NUM_VALUES, NUM_VALUES};
    Kokkos::View<float**, KokkosExecSpace> d_mb{"d_mb", NUM_VALUES, NUM_VALUES};
    Kokkos::View<float**, KokkosExecSpace> d_mc{"d_mc", NUM_VALUES, NUM_VALUES};

    auto policy = hintLightWeight(
        Kokkos::MDRangePolicy<KokkosExecSpace, Kokkos::Rank<2>>(execSpace, {{0, 0}}, {{NUM_VALUES, NUM_VALUES}}));
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) { vectorProd(d_a, d_b, d_ma, row, col); });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) { vectorProd(d_a, d_c, d_mb, row, col); });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t row, const size_t col) {
          float tmp = 0;
          for (int i = 0; i < NUM_VALUES; ++i) {
            tmp += d_ma(row, i) * d_mb(i, col);
          }
          d_mc(row, col) = tmp;
        });

    Kokkos::parallel_for(
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, NUM_VALUES)),
        KOKKOS_LAMBDA(const size_t row) {
          float tmp = 0;
          for (int i = 0; i < NUM_VALUES; ++i) {
            tmp += d_ma(row, i) * d_b[i];
          }
          d_c[row] = tmp;
        });
    return d_a;
  }
}  // namespace KOKKOS_NAMESPACE
