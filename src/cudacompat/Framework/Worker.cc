#include "Framework/Worker.h"

namespace edm {
/**
 * Initiates asynchronous data retrieval in advance of actual usage 
 * @param event reference to an event object
 * @param eventSetup constant reference to an event setup object
 * @param iTask waiting task holder object
 */
// The above comment was written by an LLM. 
  void Worker::prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask) {
    //std::cout << "prefetchAsync for " << this << " iTask " << iTask << std::endl;
    bool expected = false;
    if (prefetchRequested_.compare_exchange_strong(expected, true)) {
      //std::cout << "first prefetch call" << std::endl;
      for (Worker* dep : itemsToGet_) {
        //std::cout << "calling doWorkAsync for " << dep << " with " << iTask << std::endl;
        dep->doWorkAsync(event, eventSetup, iTask);
      }
    }
  }
}  // namespace edm
