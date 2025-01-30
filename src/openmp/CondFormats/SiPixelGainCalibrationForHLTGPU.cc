#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

/**
 * @brief Constructor initializing the object with gain data and host structure
 * @param gain reference to SiPixelGainForHLTonGPU object containing calibration parameters
 * @param gainData vector of character data used for gain calibration
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU(gain);
  gainForHLTonHost_->v_pedestals = reinterpret_cast<SiPixelGainForHLTonGPU_DecodingStructure*>(gainData_.data());
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  delete gainForHLTonHost_;
}
