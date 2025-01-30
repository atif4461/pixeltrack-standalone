#include <iostream>
#include <sstream>

#include "plugin-Validation/SimpleAtomicHisto.h"

/**
 * Main program entry point
   * Initializes SimpleAtomicHisto object with specified parameters
 * @param int number of bins in histogram
 * @param double minimum value of range
 * @param double maximum value of range
  * Fills histogram with specified value
 * @param double value to fill into histogram
  * Outputs histogram data as string stream
 * @return string representation of histogram data
 */
// The above comment was written by an LLM. 
int main() {
  SimpleAtomicHisto h(10, 0, 1);
  std::stringstream ss;

  ss << h;
  assert(ss.str() == "12 0 1 0 0 0 0 0 0 0 0 0 0 0 0");

  h.fill(-0.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 0 0 0 0 0 0 0 0 0 0 0");

  h.fill(1.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 0 0 0 0 0 0 0 0 0 0 1");

  h.fill(0);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 1 0 0 0 0 0 0 0 0 0 1");

  h.fill(0.1);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 1 1 0 0 0 0 0 0 0 0 1");

  h.fill(0.0999);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 0 0 0 0 0 0 0 0 1");

  h.fill(0.2);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 0 1");

  h.fill(0.9);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 1 1");

  h.fill(0.9999999);
  ss.str("");
  ss << h;
  assert(ss.str() == "12 0 1 1 2 1 1 0 0 0 0 0 0 2 1");

  return 0;
}
