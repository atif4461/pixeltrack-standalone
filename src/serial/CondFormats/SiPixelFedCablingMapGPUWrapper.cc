// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CMSSW includes
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

/**
 * @brief Constructor for SiPixelFedCablingMapGPUWrapper class
 * @param cablingMap Reference to SiPixelFedCablingMapGPU object
 * @param modToUnp Vector of unsigned characters representing module to unpinning
 */
// The above comment was written by an LLM. 
SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
  : modToUnpDefault(modToUnp.size()), hasQuality_(true), cablingMapHost(cablingMap) {
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

