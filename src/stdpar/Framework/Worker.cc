#include "Framework/Worker.h"

namespace edm {
/**
 * Initiates asynchronous data prefetching for dependent tasks.
 * @param event reference to the current event
 * @param eventSetup constant reference to the event setup configuration
 * @param iTask holder for waiting task management
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
