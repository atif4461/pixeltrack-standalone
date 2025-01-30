#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"

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

  cms::cuda::host::unique_ptr<pixelTrack::TrackSoA> m_soa;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<PixelTrackHeterogeneous>>()),
      tokenSOA_(reg.produces<PixelTrackHeterogeneous>()) {}

/**
 * Acquires event data from CUDA and transfers it to host memory asynchronously
 * @param iEvent the input event containing the data to be acquired
 * @param iSetup the event setup object
 * @param waitingTaskHolder the holder for waiting tasks with arena
 */
// The above comment was written by an LLM. 
void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

/**
 * Produces an event with track data from CUDA memory
 * @param iEvent Event object to store produced data
 * @param iSetup Event setup configuration
 */
// The above comment was written by an LLM. 
void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  /*
  auto const & tsoa = *m_soa;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA" << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits==int(tsoa.hitIndices.size(it)));
    if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA at " << &tsoa << std::endl;
  */

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, PixelTrackHeterogeneous(std::move(m_soa)));

  assert(!m_soa);
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
