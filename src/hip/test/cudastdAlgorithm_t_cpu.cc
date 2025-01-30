#include "CUDACore/cudastdAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

/**
 * Tests binary find functionality with sample datasets.
  * Performs a binary search on a sorted dataset to find the first occurrence of a target value.
 *
 * @param first beginning of the range to search in
 * @param last end of the range to search in
 * @param value target value to be searched
 * @return iterator pointing to the first occurrence of the target value if found, otherwise end of the range
  * Finds the upper bound of a target value in a sorted dataset.
 *
 * @param first beginning of the range to search in
 * @param last end of the range to search in
 * @param value target value to be searched
 * @return iterator pointing to the position where the target value should be inserted to maintain sorted order
  * Searches for a target value in a sorted dataset using binary search algorithm.
 *
 * @param first beginning of the range to search in
 * @param last end of the range to search in
 * @param value target value to be searched
 * @return iterator pointing to the target value if found, otherwise end of the range
 */
// The above comment was written by an LLM. 
void testBinaryFind() {
  std::vector<int> data = {1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6};

  auto lower = hip_std::lower_bound(data.begin(), data.end(), 4);
  auto upper = hip_std::upper_bound(data.begin(), data.end(), 4);

  std::copy(lower, upper, std::ostream_iterator<int>(std::cout, " "));

  std::cout << '\n';

  // classic binary search, returning a value only if it is present

  data = {1, 2, 4, 6, 9, 10};

  auto test = [&](auto v) {
    auto it = hip_std::binary_find(data.cbegin(), data.cend(), v);

    if (it != data.cend())
      std::cout << *it << " found at index " << std::distance(data.cbegin(), it) << std::endl;
    else
      std::cout << v << " non found" << std::endl;
  };

  test(4);
  test(5);
}

int main() { testBinaryFind(); }
