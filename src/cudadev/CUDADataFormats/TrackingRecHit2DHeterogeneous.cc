#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

template <>
/**
 * Returns a host unique pointer to an array of floats representing local coordinates
 * @return float array containing local coordinates
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
/**
 * @brief Asynchronously transfers module start hits from device to host
 * @param stream CUDA stream used for asynchronous transfer
 * @return Unique pointer to array of module start hits on host
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(gpuClustering::maxNumModules + 1, stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_hitsModuleStart, sizeof(uint32_t) * (gpuClustering::maxNumModules + 1), cudaMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * @brief Copies global coordinates from device memory to host memory asynchronously
 * @param stream CUDA stream for asynchronous operation
 * @return Unique pointer to array of floats containing copied coordinates
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
 * @brief Copies charges from device memory to host memory asynchronously
 * @param stream CUDA stream to perform operation on
 * @return Unique pointer to array of int32_t containing charge data on host
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
 * @brief Copies size data from device to host asynchronously
 * @param stream CUDA stream for asynchronous operation
 * @return Unique pointer to host array containing size data
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaCheck(cudaMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), cudaMemcpyDefault, stream));
  return ret;
}
