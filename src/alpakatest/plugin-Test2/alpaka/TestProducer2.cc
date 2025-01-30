#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "alpakaAlgo2.h"

namespace {
  std::atomic<int> nevents;
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer2 : public edm::EDProducerExternalWork {
  public:
    explicit TestProducer2(edm::ProductRegistry& reg);

  private:
    void acquire(edm::Event const& event,
                 edm::EventSetup const& eventSetup,
                 edm::WaitingTaskWithArenaHolder holder) override;
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
    void endJob() override;

    edm::EDGetTokenT<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>> getToken_;
  };

/**
 * Constructor initializing the test producer with product registry
 * @param reg Product registry reference for consuming products
 */
// The above comment was written by an LLM. 
  TestProducer2::TestProducer2(edm::ProductRegistry& reg)
      : getToken_(reg.consumes<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>>()) {
    nevents = 0;
  }

/**
 * Acquires an event from the data source and processes it.
 * @param event The input event to be processed.
 * @param eventSetup The setup associated with the event.
 * @param holder The waiting task holder for the acquisition process.
 */
// The above comment was written by an LLM. 
  void TestProducer2::acquire(edm::Event const& event,
                              edm::EventSetup const& eventSetup,
                              edm::WaitingTaskWithArenaHolder holder) {
    auto const& tmp = event.get(getToken_);
    cms::alpakatools::ScopedContextAcquire<Queue> ctx(tmp, std::move(holder));
    auto const& array = ctx.get(tmp);
    alpakaAlgo2(ctx.stream());

    std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << " array "
              << array.data() << std::endl;
  }

/**
 * @brief Produces data for an event in the TestProducer2 module
 * @param event Reference to the current event being processed
 * @param eventSetup Constant reference to the event setup configuration
 */
// The above comment was written by an LLM. 
  void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;
    ++nevents;
  }

  void TestProducer2::endJob() {
    std::cout << "TestProducer2::endJob processed " << nevents.load() << " events" << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer2);
