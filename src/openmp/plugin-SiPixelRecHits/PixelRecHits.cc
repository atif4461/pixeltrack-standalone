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
 * Sets the start of the hits layer based on the provided module starts and geometry parameters.
 * @param[in] hitsModuleStart array of module starts
 * @param[in] cpeParams constant pointer to pixelCPEforGPU parameters
 * @param[out] hitsLayerStart array to store the calculated layer starts
 */
// The above comment was written by an LLM. 
   void setHitsLayerStart(uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) {
    assert(0 == hitsModuleStart[0]);

    int begin = 0;
    constexpr int end = 11;
    for (int i = begin; i < end; i += 1) {
      hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
      printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
    }
  }
}  // namespace

namespace pixelgpudetails {

/**
 * @brief Creates tracking rec hits in 2D CPU from given digis, clusters and beam spot
 * @param[in] digis_d SiPixelDigisSoA object containing digital information
 * @param[in] clusters_d SiPixelClustersSoA object containing cluster information
 * @param[in] bs_d BeamSpotPOD object containing beam spot information
 * @param[in] cpeParams pointer to pixelCPEforGPU params object
 * @return TrackingRecHit2DCPU object containing created hits
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

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      setHitsLayerStart(clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());
    }

    if (nHits) {
      cms::cuda::fillManyFromVector(hits_d.phiBinner(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits);
    }

    return hits_d;
  }

}  // namespace pixelgpudetails
