#include <hip/hip_runtime.h>

#include "CUDACore/Product.h"
#include "CUDACore/HostProduct.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "CUDACore/ScopedContext.h"

class PixelVertexSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit PixelVertexSoAFromCUDA(edm::ProductRegistry& reg);
  ~PixelVertexSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::hip::Product<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  cms::hip::host::unique_ptr<ZVertexSoA> m_soa;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::hip::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

/**
 * Acquires data from an event and sets up the CUDA context
 * @param iEvent the input event containing the data to acquire
 * @param iSetup the event setup used to configure the acquisition process
 * @param waitingTaskHolder holder for the waiting task with arena
 */
// The above comment was written by an LLM. 
void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::hip::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

/**
 * Produces pixel vertex data in StructureOfArrays format 
 */
// The above comment was written by an LLM. 
void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
