#include <utility>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "AlpakaDataFormats/ZVertexHost.h"
#include "AlpakaDataFormats/alpaka/ZVertexAlpaka.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"
#include "Framework/RunningAverage.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelVertexSoAFromAlpaka : public edm::EDProducerExternalWork {
  public:
    explicit PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg);
    ~PixelVertexSoAFromAlpaka() override = default;

  private:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    edm::EDGetTokenT<cms::alpakatools::Product<Queue, ZVertexAlpaka>> tokenDevice_;
    edm::EDPutTokenT<ZVertexHost> tokenHost_;

    ZVertexHost soa_;
  };

/**
 * Constructor for initializing PixelVertexSoAFromAlpaka object with ProductRegistry reference
 */
// The above comment was written by an LLM. 
  PixelVertexSoAFromAlpaka::PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenDevice_(reg.consumes<cms::alpakatools::Product<Queue, ZVertexAlpaka>>()),
        tokenHost_(reg.produces<ZVertexHost>()),
        soa_(cms::alpakatools::make_host_buffer<ZVertexSoA, Platform>()) {}

/**
 * Acquires event data from the input stream and stores it in a host buffer
 */
// The above comment was written by an LLM. 
  void PixelVertexSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenDevice_);
    cms::alpakatools::ScopedContextAcquire<Queue> ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    soa_ = cms::alpakatools::make_host_buffer<ZVertexSoA>(ctx.stream());
    alpaka::memcpy(ctx.stream(), soa_, inputData);
  }

/**
 * Produces PixelVertex data from Alpaka and stores it in the event
 * @param iEvent reference to the Event object being processed
 * @param iSetup constant reference to the EventSetup object
 */
// The above comment was written by an LLM. 
  void PixelVertexSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    // No copies....
    iEvent.emplace(tokenHost_, std::move(soa_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexSoAFromAlpaka);
