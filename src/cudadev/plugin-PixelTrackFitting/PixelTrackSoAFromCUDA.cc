#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"

// Switch on to enable checks and printout for found tracks
#undef PIXEL_DEBUG_PRODUCE

class PixelTrackSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit PixelTrackSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelTrackSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> soa_;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<PixelTrackHeterogeneous>>()),
      tokenSOA_(reg.produces<PixelTrackHeterogeneous>()) {}

/**
 * Acquires data from an event and setup, processing it with CUDA functionality
 * @param iEvent the input event containing data to be processed
 * @param iSetup the event setup used in data acquisition
 * @param waitingTaskHolder holder for tasks waiting for completion
 */
// The above comment was written by an LLM. 
void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  soa_ = inputData.toHostAsync(ctx.stream());
}

/**
 * Produces PixelTrack data from CUDA and stores it in the event.
 *
 * @param iEvent The event where the produced data will be stored.
 * @param iSetup The event setup object.
 */
// The above comment was written by an LLM. 
void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
#ifdef PIXEL_DEBUG_PRODUCE
  auto const& tsoa = *soa_;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA" << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits == int(tsoa.hitIndices.size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA at " << &tsoa << std::endl;
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, PixelTrackHeterogeneous(std::move(soa_)));

  assert(!soa_);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
