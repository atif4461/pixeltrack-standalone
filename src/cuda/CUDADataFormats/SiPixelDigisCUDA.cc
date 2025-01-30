#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"

/**
 * Constructor initializing device memory for pixel digis data structures
 * @param maxFedWords maximum number of words in a FED buffer
 * @param stream CUDA stream object
 */
// The above comment was written by an LLM. 
SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream) {
  xx_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
}

/**
 * @brief Returns a host unique pointer to an array of uint16_t values 
 *        representing pixel digis transferred from device to host asynchronously
 * @param stream cuda stream used for asynchronous transfer
 * @return host unique pointer to an array of uint16_t values
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Returns a host unique pointer to an array of cluster data transferred from device memory asynchronously
 * @param stream CUDA stream used for asynchronous transfer
 * @return Host unique pointer to an array of int32_t values
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

/**
 * Returns a host unique pointer to an array of uint32_t containing pixel digis
 * asynchronously transferred from device memory to host memory using the specified stream
 * @return unique pointer to host array of pixel digis
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

/**
 * Returns a host unique pointer to an array of uint32_t containing digi IDs
 * @return host unique pointer to uint32_t array
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
