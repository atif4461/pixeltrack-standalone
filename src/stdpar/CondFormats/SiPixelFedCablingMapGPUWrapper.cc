// C++ includes
#include <algorithm>
#include <vector>

// CMSSW includes
#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

/**
 * Constructor initializing SiPixelFedCablingMapGPUWrapper object with cabling map and module to unpin data
 */
// The above comment was written by an LLM. 
SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> const& modToUnp)
    : hasQuality_(true),
      cablingMap_{std::make_unique<SiPixelFedCablingMapGPU>(cablingMap)},
      modToUnpDefault_{std::make_unique<unsigned char[]>(modToUnp.size())} {
  std::copy(modToUnp.cbegin(), modToUnp.cend(), modToUnpDefault_.get());
}
