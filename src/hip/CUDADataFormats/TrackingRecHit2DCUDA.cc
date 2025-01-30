#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

template <>
/**
 * @brief Returns local coordinates of tracking rec hits as host unique pointer array
 * @return Unique pointer to float array containing local coordinates
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<float[]>(4 * nHits(), stream);
  cms::hip::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
/**
 * @brief Returns unique pointer to array of hit module start values transferred from device to host asynchronously
 * @param stream hipStream_t object representing the stream for asynchronous transfer
 * @return Unique pointer to array of uint32_t values
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(hipMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, hipMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies global coordinates from device memory to host memory asynchronously.
 * @param stream Hip stream used for asynchronous operation
 * @return Unique pointer to array of floats representing global coordinates on host
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<float[]>(4 * nHits(), stream);
  cudaCheck(
      hipMemcpyAsync(ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), hipMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies charge data from device to host asynchronously
 * @param stream CUDA stream for asynchronous operation
 * @return Unique pointer to host array containing charge data
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<int32_t[]>(nHits(), stream);
  cudaCheck(
      hipMemcpyAsync(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), hipMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies size data from device to host asynchronously
 * @param stream hipStream_t object representing the stream to perform operation on
 * @return unique pointer to an array of int16_t values on the host
 */
// The above comment was written by an LLM. 
cms::hip::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(hipStream_t stream) const {
  auto ret = cms::hip::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaCheck(hipMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), hipMemcpyDefault, stream));
  return ret;
}
