#include "DataFormats/SiPixelDigisSoA.h"

#include <cassert>

/**
 * Constructor initializing SiPixelDigisSoA object with digi data
 * @param nDigis number of digis
 * @param pdigi array of digi values
 * @param rawIdArr array of raw identifier values
 * @param adc array of analog to digital converter values
 * @param clus array of cluster values
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
