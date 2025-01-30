// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

/**
 * Constructor initializing SiPixelFedCablingMapGPUWrapper object with given cabling map and module to unpinning data.
 */
// The above comment was written by an LLM. 
SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> const& modToUnp)
    : hasQuality_(true) {
  cudaCheck(cudaMallocManaged(&cablingMap_, sizeof(SiPixelFedCablingMapGPU)));
  *cablingMap_ = cablingMap;
  cudaCheck(cudaMallocManaged(&modToUnpDefault_, modToUnp.size()));
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault_);
  for (int device = 0, ndev = cms::cuda::deviceCount(); device < ndev; ++device) {
#ifndef CUDAUVM_DISABLE_ADVISE
    cudaCheck(cudaMemAdvise(cablingMap_, sizeof(SiPixelFedCablingMapGPU), cudaMemAdviseSetReadMostly, device));
    cudaCheck(cudaMemAdvise(modToUnpDefault_, sizeof(modToUnp.size()), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
    cms::cuda::ScopedSetDevice guard{device};
    auto stream = cms::cuda::getStreamCache().get();
    cudaCheck(cudaMemPrefetchAsync(cablingMap_, sizeof(SiPixelFedCablingMapGPU), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(modToUnpDefault_, modToUnp.size(), device, stream.get()));
#endif
  }
}

/**
 * Destructor to release GPU memory allocated for cabling map and module to unpin default data structures */
// The above comment was written by an LLM. 
SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {
  cudaCheck(cudaFree(cablingMap_));
  cudaCheck(cudaFree(modToUnpDefault_));
}
