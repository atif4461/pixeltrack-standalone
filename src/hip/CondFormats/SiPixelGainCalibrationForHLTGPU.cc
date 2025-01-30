#include <hip/hip_runtime.h>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cudaCheck.h"

/**
 * Constructor initializing SiPixelGainCalibrationForHLTGPU object with gain and gain data
 * @param gain reference to SiPixelGainForHLTonGPU object containing gain values
 * @param gainData vector of characters representing gain data
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  cudaCheck(hipHostMalloc(&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() { cudaCheck(hipHostFree(gainForHLTonHost_)); }

/**
 * Destructor to free GPU memory allocated for gain data and HLT data
 */
// The above comment was written by an LLM. 
SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cudaCheck(hipFree(gainForHLTonGPU));
  cudaCheck(hipFree(gainDataOnGPU));
}

/**
 * Returns a pointer to the GPU product asynchronously.
 *
 * @param[in] cudaStream hipStream_t object representing the CUDA stream.
 * @return Pointer to the gain for HL Ton GPU structure.
 */
// The above comment was written by an LLM. 
const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(hipStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, hipStream_t stream) {
    cudaCheck(hipMalloc((void**)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
    cudaCheck(hipMalloc((void**)&data.gainDataOnGPU, this->gainData_.size()));
    // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
    cudaCheck(
        hipMemcpyAsync(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size(), hipMemcpyDefault, stream));

    cudaCheck(hipMemcpyAsync(
        data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), hipMemcpyDefault, stream));
    cudaCheck(hipMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals),
                             &(data.gainDataOnGPU),
                             sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                             hipMemcpyDefault,
                             stream));
  });
  return data.gainForHLTonGPU;
}
