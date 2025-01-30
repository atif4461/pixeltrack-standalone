#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "alpakaAlgo1.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer : public edm::EDProducer {
  public:
    explicit TestProducer(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>> putToken_;
  };

  TestProducer::TestProducer(edm::ProductRegistry& reg)
      : rawGetToken_(reg.consumes<FEDRawDataCollection>()),
        putToken_(reg.produces<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>>()) {}

/**
 * Produces data for an event in the framework of a particle physics experiment 
  
 * Constructs a scoped context for producing data within a specific stream 
 * and executes the algorithm associated with it 
 */
// The above comment was written by an LLM. 
  void TestProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const value = event.get(rawGetToken_).FEDData(1200).size();
    std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << " FED 1200 size " << value << std::endl;

    cms::alpakatools::ScopedContextProduce<Queue> ctx(event.streamID());
    ctx.emplace(event, putToken_, alpakaAlgo1(ctx.stream()));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer);
