#ifdef GPU_DEBUG
#include <iostream>
#endif

#include "AlpakaCore/AllocatorPolicy.h"
#include "AlpakaCore/config.h"
#include "CondFormats/pixelCPEforGPU.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    struct setHitsLayerStart {
      template <typename TAcc>
/**
 * @brief Applies the layer start indices calculation
 * @param[in] acc Accelerator object
 * @param[in] hitsModuleStart Array of module start indices
 * @param[in] cpeParams Parameters for pixel CPE on GPU
 * @param[out] hitsLayerStart Array to store calculated layer start indices
 */
// The above comment was written by an LLM. 
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    uint32_t const* __restrict__ hitsModuleStart,
                                    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                    uint32_t* hitsLayerStart) const {
        ALPAKA_ASSERT_OFFLOAD(0 == hitsModuleStart[0]);

        cms::alpakatools::for_each_element_in_grid(acc, 11, [&](uint32_t i) {
          hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
          printf("LayerStart %d %d: %d\n", i, cpeParams->layerGeometry().layerStart[i], hitsLayerStart[i]);
#endif
        });
      }
    };
  }  // namespace

  namespace pixelgpudetails {

/**
 * @brief Creates tracking rec hits asynchronously
 * @param[in] digis_d SiPixelDigisAlpaka object containing digitized data
 * @param[in] clusters_d SiPixelClustersAlpaka object containing cluster data
 * @param[in] bs_d BeamSpotAlpaka object containing beam spot data
 * @param[in] cpeParams pointer to pixelCPEforGPU parameters
 * @param[out] queue Queue object for asynchronous operations
 * @return TrackingRecHit2DAlpaka object containing generated hits
 */
// The above comment was written by an LLM. 
    TrackingRecHit2DAlpaka PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisAlpaka const& digis_d,
                                                               SiPixelClustersAlpaka const& clusters_d,
                                                               BeamSpotAlpaka const& bs_d,
                                                               pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                               Queue& queue) const {
      auto nHits = clusters_d.nClusters();
      TrackingRecHit2DAlpaka hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), queue);

      const int threadsPerBlockOrElementsPerThread = 128;
      const int blocks = digis_d.nModules();  // active modules (with digis)
      const auto getHitsWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlockOrElementsPerThread);

#ifdef GPU_DEBUG
      std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
      if (blocks) {  // protect from empty events
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(getHitsWorkDiv,
                                                        gpuPixelRecHits::getHits(),
                                                        cpeParams,
                                                        bs_d.data(),
                                                        digis_d.view(),
                                                        digis_d.nDigis(),
                                                        clusters_d.view(),
                                                        hits_d.view()));
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif

      // assuming full warp of threads is better than a smaller number...
      if (nHits) {
        const auto oneBlockWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
        alpaka::enqueue(
            queue,
            alpaka::createTaskKernel<Acc1D>(
                oneBlockWorkDiv, setHitsLayerStart(), clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart()));
      }

      if (nHits) {
        cms::alpakatools::fillManyFromVector<Acc1D>(
            hits_d.phiBinner(), 10, hits_d.c_iphi(), hits_d.c_hitsLayerStart(), nHits, 256, queue);
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif

#if defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND && defined ALPAKA_DISABLE_CACHING_ALLOCATOR
      // FIXME this is required to keep the host buffer inside hits_d alive; it could be removed once the host buffers are also stream-ordered
      alpaka::wait(queue);
#endif

      return hits_d;
    }

  }  // namespace pixelgpudetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
