#include "CUDACore/cudastdAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

/**
 * @brief Tests binary find functionality with example usage.
 
  * @brief Finds the range of elements equal to a specified value in a sorted sequence.
 * @param first Beginning of the range to search.
 * @param last End of the range to search.
 * @param val Value to search for.
 * @return Iterator pointing to the first element that is not less than val.
 
  * @brief Finds the range of elements equal to a specified value in a sorted sequence.
 * @param first Beginning of the range to search.
 * @param last End of the range to search.
 * @param val Value to search for.
 * @return Iterator pointing to the first element greater than val.
 
  * @brief Performs a classic binary search for an exact match in a sorted sequence.
 * @param first Beginning of the range to search.
 * @param last End of the range to search.
 * @param val Value to search for.
 * @return Iterator pointing to the found element if it exists, end iterator otherwise.
 */
// The above comment was written by an LLM. 
void testBinaryFind() {
  std::vector<int> data = {1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6};

  auto lower = cuda_std::lower_bound(data.begin(), data.end(), 4);
  auto upper = cuda_std::upper_bound(data.begin(), data.end(), 4);

  std::copy(lower, upper, std::ostream_iterator<int>(std::cout, " "));

  std::cout << '\n';

  // classic binary search, returning a value only if it is present

  data = {1, 2, 4, 6, 9, 10};

  auto test = [&](auto v) {
    auto it = cuda_std::binary_find(data.cbegin(), data.cend(), v);

    if (it != data.cend())
      std::cout << *it << " found at index " << std::distance(data.cbegin(), it) << std::endl;
    else
      std::cout << v << " non found" << std::endl;
  };

  test(4);
  test(5);
}

int main() { testBinaryFind(); }
