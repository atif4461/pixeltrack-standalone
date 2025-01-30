// C++ headers
#include <algorithm>
#include <numeric>

// CMSSW headers
#include "CUDACore/cudaCompat.h"

#include "plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.h"  // !
#include "plugin-SiPixelClusterizer/gpuClusteringConstants.h"        // !

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
/**
 * Sets the layer start values in the hitsLayerStart array based on the input parameters.
 * @param[in] hitsModuleStart pointer to an array of hit module start indices
 * @param[in] cpeParams pointer to constants and parameters for pixel CPE calculation on GPU
 * @param[out] hitsLayerStart pointer to an array where layer start indices will be stored
 */
// The above comment was written by an LLM. 
  __global__ void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    assert(0 == hitsModuleStart[0]);

    int begin = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int end = 11;
    for (int i = begin; i < end; i += blockDim.x * gridDim.x) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

/**
 * @brief Creates 2D CPU tracking rec hits from pixel digis and clusters
 * @param[in] digis_d input digis in SoA format
 * @param[in] clusters_d input clusters in SoA format
 * @param[in] bs_d beam spot object
 * @param[in] cpeParams parameters for GPU cluster parameter estimation
 * @return vector of 2D CPU tracking rec hits
 */
// The above comment was written by an LLM. 
  TrackingRecHit2DCPU PixelRecHitGPUKernel::makeHits(SiPixelDigisSoA const& digis_d,
                                                     SiPixelClustersSoA const& clusters_d,
                                                     BeamSpotPOD const& bs_d,
                                                     pixelCPEforGPU::ParamsOnGPU const* cpeParams) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCPU hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), nullptr);

    if (digis_d.nModules())  // protect from empty events
      gpuPixelRecHits::getHits(cpeParams, &bs_d, digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
    cudaCheck(cudaGetLastError());

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      setHitsLayerStart(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256);
    }

    return hits_d;
  }

}  // namespace pixelgpudetails
