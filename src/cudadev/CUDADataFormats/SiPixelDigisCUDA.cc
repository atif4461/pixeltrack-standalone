#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

/**
 * Constructor initializing member variables with device unique pointers 
 * for pixel digi data and creating a host view of device constants.
 */
// The above comment was written by an LLM. 
SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : xx_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      yy_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      adc_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      moduleInd_d(cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream)),
      clus_d(cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream)),
      view_d(cms::cuda::make_device_unique<DeviceConstView>(stream)),
      pdigi_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream)),
      rawIdArr_d(cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream)) {
  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  cms::cuda::copyAsync(view_d, view, stream);
}

/**
 * Returns a host unique pointer to an array of uint16_t values 
 * representing pixel digis after asynchronous transfer from device memory
 * @return host unique pointer to array of uint16_t values
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

/**
 * Returns a host unique pointer to an array of int32_t containing cluster data
 * transferred from device memory asynchronously
 *
 * @return host unique pointer to array of int32_t
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Copies pixel digi data from device to host asynchronously
 * @param stream CUDA stream for asynchronous operation
 * @return Unique pointer to host array containing pixel digi data
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Returns a host unique pointer of uint32_t array containing digi IDs asynchronously
 * @param stream CUDA stream object used for asynchronous operations
 * @return Host unique pointer of uint32_t array
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
