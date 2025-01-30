#include <cuda.h>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cudaCheck.h"

/**
 * Constructor initializing the object with gain data and calibration parameters
 * @param gain reference to SiPixelGainForHLTonGPU object containing gain calibration data
 * @param gainData vector of character data used for gain calibration
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  /*
  cudaCheck(cudaMallocHost(&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  *gainForHLTonHost_ = gain;
  */
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU(gain);
  gainForHLTonHost_->v_pedestals = reinterpret_cast<SiPixelGainForHLTonGPU_DecodingStructure*>(gainData_.data());
}

/**
 * Destructor to free allocated resources on host 
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  /*
  cudaCheck(cudaFreeHost(gainForHLTonHost_));
  */
  delete gainForHLTonHost_;
}

/**
 * Destructor to free GPU memory allocated for gain data and HLT data
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cudaCheck(cudaFree(gainForHLTonGPU));
  cudaCheck(cudaFree(gainDataOnGPU));
}

/**
 * Returns pointer to GPU product asynchronously 
 */
// The above comment was written by an LLM. 
const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    cudaCheck(cudaMalloc((void**)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
    cudaCheck(cudaMalloc((void**)&data.gainDataOnGPU, this->gainData_.size()));
    // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
    cudaCheck(
        cudaMemcpyAsync(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size(), cudaMemcpyDefault, stream));

    cudaCheck(cudaMemcpyAsync(
        data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals),
                              &(data.gainDataOnGPU),
                              sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.gainForHLTonGPU;
}
