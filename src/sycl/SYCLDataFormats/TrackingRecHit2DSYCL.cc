#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

/**
 * @brief Copies local coordinates from device memory to host memory asynchronously
 * @param stream SYCL queue used for asynchronous data transfer
 * @return Unique pointer to array of floats containing copied coordinates
 */
// The above comment was written by an LLM. 
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DSYCL::localCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get(), 4 * nHits() * sizeof(float));
  return ret;
}

/**
 * @brief Copies module start hits from device to host asynchronously
 * @param stream SYCL queue object used for asynchronous data transfer
 * @return Unique pointer to array of uint32_t values representing module start hits on host
 */
// The above comment was written by an LLM. 
cms::sycltools::host::unique_ptr<uint32_t[]> TrackingRecHit2DSYCL::hitsModuleStartToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(2001, stream);
  stream.memcpy(ret.get(), m_hitsModuleStart, 4 * 2001 * sizeof(uint32_t));
  return ret;
}

/**
 * @brief Copies global coordinates from device memory to host memory asynchronously
 * @param stream SYCL queue used for asynchronous data transfer
 * @return unique pointer to array of floats containing global coordinates on host
 */
// The above comment was written by an LLM. 
cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DSYCL::globalCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float));
  return ret;
}

/**
 * @brief Copies charge data from device memory to host memory asynchronously
 * @param stream SYCL queue object used for asynchronous data transfer
 * @return Unique pointer to an array of 32bit integers representing charge data on the host
 */
// The above comment was written by an LLM. 
cms::sycltools::host::unique_ptr<int32_t[]> TrackingRecHit2DSYCL::chargeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t));
  return ret;
}

/**
 * @brief Copies size data from device memory to host memory asynchronously
 * @param stream SYCL queue object used for asynchronous operation
 * @return Unique pointer to host array containing size data
 */
// The above comment was written by an LLM. 
cms::sycltools::host::unique_ptr<int16_t[]> TrackingRecHit2DSYCL::sizeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int16_t[]>(2 * nHits(), stream);
  stream.memcpy(ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t));
  return ret;
}
