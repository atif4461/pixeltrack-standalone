#include <algorithm>
#include <memory>
#include <vector>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

/**
 * Constructor initializing SiPixelGainCalibrationForHLTGPU object with gain and gain data
 * @param gain reference to SiPixelGainForHLTonGPU object containing gain values
 * @param gainData vector of DecodingStructure objects containing gain calibration data
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<DecodingStructure> const& gainData)
    : gainForHLT_{std::make_unique<SiPixelGainForHLTonGPU>(gain)},
      gainData_{std::make_unique<DecodingStructure[]>(gainData.size())} {
  gainForHLT_->v_pedestals = gainData_.get();
  std::copy(gainData.cbegin(), gainData.cend(), gainData_.get());
}
