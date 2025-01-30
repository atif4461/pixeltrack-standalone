#include <cuda_runtime.h>

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

  edm::EDGetTokenT<cms::cuda::Product<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  cms::cuda::host::unique_ptr<ZVertexSoA> m_soa;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(edm::ProductRegistry& reg)
    : tokenCUDA_(reg.consumes<cms::cuda::Product<ZVertexHeterogeneous>>()),
      tokenSOA_(reg.produces<ZVertexHeterogeneous>()) {}

/**
 * Acquires event data from CUDA and transfers it to host memory asynchronously
 */
// The above comment was written by an LLM. 
void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

/**
 * Produces PixelVertexSoA data and adds it to the event
 * @param iEvent the event to which the data is added
 * @param iSetup the event setup object
 */
// The above comment was written by an LLM. 
void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
