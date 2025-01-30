#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

template <>
/**
 * @brief Returns a host unique pointer to an array of floats representing local coordinates.
 * @param stream CUDA stream used for asynchronous operations.
 * @return A host unique pointer to an array of floats.
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
/**
 * @brief Returns an array of module start hits transferred from device to host asynchronously
 * @param stream CUDA stream for asynchronous memory transfer
 * @return Unique pointer to an array of uint32_t values representing module start hits
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies global coordinates from device memory to host memory asynchronously.
 * @param stream CUDA stream used for asynchronous operation
 * @return Unique pointer to array of floats containing global coordinates on host
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), cudaMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies charge data from device to host asynchronously
 * @param stream CUDA stream for asynchronous operation
 * @return unique pointer to host array containing charge data
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nHits(), stream);
  cudaCheck(
      cudaMemcpyAsync(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), cudaMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies size data from device to host asynchronously.
 *
 * @param stream CUDA stream to perform operation on
 * @return Unique pointer to array of int16_t values representing size data
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), cudaMemcpyDefault, stream));
  return ret;
}
