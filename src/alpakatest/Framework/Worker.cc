//#include <iostream>

#include "Framework/Worker.h"

namespace edm {
/**
 * Initiates asynchronous data retrieval for dependent tasks in preparation for processing.
 * @param event Reference to the current event being processed.
 * @param eventSetup Constant reference to the setup for the current event.
 * @param iTask Holder for the task waiting to be executed.
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
