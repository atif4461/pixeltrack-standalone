#include "DataFormats/SiPixelDigisSoA.h"

#include <cassert>

/**
 * Constructor initializing pixel digi data structures with input parameters
 * @param nDigis number of digis
 * @param pdigi pointer to digi data array
 * @param rawIdArr pointer to raw ID array
 * @param adc pointer to ADC values array
 * @param clus pointer to cluster data array
 */
// The above comment was written by an LLM. 
SiPixelDigisSoA::SiPixelDigisSoA(
    size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus)
    : pdigi_(pdigi, pdigi + nDigis),
      rawIdArr_(rawIdArr, rawIdArr + nDigis),
      adc_(adc, adc + nDigis),
      clus_(clus, clus + nDigis) {
  assert(pdigi_.size() == nDigis);
}
