#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"

class CAHitNtupletCUDA : public edm::EDProducer {
public:
  explicit CAHitNtupletCUDA(edm::ProductRegistry& reg);
  ~CAHitNtupletCUDA() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenTrackCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

/**
 * Constructor for CAHitNtupletCUDA class 
 * initializes dependencies from ProductRegistry 
 */
// The above comment was written by an LLM. 
CAHitNtupletCUDA::CAHitNtupletCUDA(edm::ProductRegistry& reg)
    : tokenHitCPU_{reg.consumes<TrackingRecHit2DCPU>()},
      tokenTrackCPU_{reg.produces<PixelTrackHeterogeneous>()},
      gpuAlgo_(reg) {}

/**
 * Produce track candidates from hit collections in an event
 * @param[in] iEvent Event object containing input data
 * @param[in] es Event setup object providing configuration
 */
// The above comment was written by an LLM. 
void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  auto bf = 0.0114256972711507;  // 1/fieldInGeV

  auto const& hits = iEvent.get(tokenHitCPU_);

  iEvent.emplace(tokenTrackCPU_, gpuAlgo_.makeTuples(hits, bf));
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
