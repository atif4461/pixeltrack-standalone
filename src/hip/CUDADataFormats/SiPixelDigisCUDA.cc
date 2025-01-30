#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"

/**
 * Constructor initializing device memory for pixel digis data structures
 * @param maxFedWords maximum number of words in a single FED fragment
 * @param stream HIP stream object for asynchronous operations
 */
// The above comment was written by an LLM. 
SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, hipStream_t stream) {
  xx_d = cms::hip::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::hip::make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d = cms::hip::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::hip::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::hip::make_device_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::hip::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::hip::make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cms::hip::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::hip::make_device_unique<DeviceConstView>(stream);
  cms::hip::copyAsync(view_d, view, stream);
}

/**
 * @brief Copies ADC data from device to host asynchronously
 * @param stream hipStream object used for asynchronous operation
 * @return unique pointer to array of uint16_t containing copied ADC data
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::hip::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Copies cluster data from device memory to host memory asynchronously
 * @param stream hipStream object used for asynchronous operation
 * @return unique pointer to array of integers containing copied cluster data
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::hip::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Copies pixel digi data from device memory to host memory asynchronously
 * @return Unique pointer to host array containing pixel digi data
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::hip::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

/**
 * @brief Copies the raw ID array from device memory to host memory asynchronously.
 *
 * @param stream hipStream_t object representing the CUDA stream
 * @return unique pointer to an array of uint32_t values on the host
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::hip::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
