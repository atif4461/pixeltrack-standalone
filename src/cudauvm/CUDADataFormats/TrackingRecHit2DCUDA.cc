#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

#ifdef CUDAUVM_DISABLE_MANAGED_RECHIT
template <>
/**
 * Returns a host unique pointer to an array of floats containing local coordinates
 * @return float array with shape 4xnHits
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
/**
 * @brief Returns a host unique pointer to an array of uint32_t values representing module start hits transferred from device to host asynchronously.
 * @param stream CUDA stream used for asynchronous memory transfer
 * @return Unique pointer to an array of uint32_t values
 */
// The above comment was written by an LLM. 
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream));
  return ret;
}

template <>
/**
 * Returns a host unique pointer to an array of floats containing global coordinates
 * of all tracking rec hits in asynchronous manner on specified CUDA stream
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
 * @return Unique pointer to host array containing charge data
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
#else
template <>
/**
 * Prefetches data from device memory to host memory asynchronously.
 * @param device   the ID of the device to prefetch from
 * @param stream   the CUDA stream to use for the prefetch operation
 */
// The above comment was written by an LLM. 
void TrackingRecHit2DCUDA::localCoordToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get(), 4 * nHits(), device, stream));
#endif
}
template <>
/**
 * Prefetches data from device memory to host memory asynchronously 
 */
// The above comment was written by an LLM. 
void TrackingRecHit2DCUDA::hitsModuleStartToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_hitsModuleStart, 4 * 2001, device, stream));
#endif
}
template <>
/**
 * Transfers data from global memory to the host in an asynchronous manner 
 */
// The above comment was written by an LLM. 
void TrackingRecHit2DCUDA::globalCoordToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), device, stream));
#endif
}
template <>
/**
 * Prefetches charge data from device memory to host memory asynchronously
 */
// The above comment was written by an LLM. 
void TrackingRecHit2DCUDA::chargeToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), device, stream));
#endif
}
template <>
/**
 * Prefetches data from device memory to host memory asynchronously 
 */
// The above comment was written by an LLM. 
void TrackingRecHit2DCUDA::sizeToHostPrefetchAsync(int device, cudaStream_t stream) const {
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), device, stream));
#endif
}

#endif
