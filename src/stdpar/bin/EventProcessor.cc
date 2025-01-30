#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
/**
 * @brief Constructor for EventProcessor class
 * @param maxEvents maximum number of events to process
 * @param runForMinutes duration in minutes to run the processor
 * @param numberOfStreams number of streams to handle
 * @param path vector of strings representing paths
 * @param esproducers vector of strings representing event setup producers
 * @param datadir file system path for data directory
 * @param validation boolean flag for validation mode
 */
// The above comment was written by an LLM. 
  EventProcessor::EventProcessor(int maxEvents,
                                 int runForMinutes,
                                 int numberOfStreams,
                                 std::vector<std::string> const& path,
                                 std::vector<std::string> const& esproducers,
                                 std::filesystem::path const& datadir,
                                 bool validation)
      : source_(maxEvents, runForMinutes, registry_, datadir, validation) {
    for (auto const& name : esproducers) {
      pluginManager_.load(name);
      auto esp = ESPluginFactory::create(name, datadir);
      esp->produce(eventSetup_);
    }

    //schedules_.reserve(numberOfStreams);
    for (int i = 0; i < numberOfStreams; ++i) {
      schedules_.emplace_back(registry_, pluginManager_, &source_, &eventSetup_, i, path);
    }
  }

/**
 * Runs event processing to completion waiting for all tasks to finish 
 */
// The above comment was written by an LLM. 
  void EventProcessor::runToCompletion() {
    source_.startProcessing();
    // The task that waits for all other work
    FinalWaitingTask globalWaitTask;
    tbb::task_group group;
    for (auto& s : schedules_) {
      s.runToCompletionAsync(WaitingTaskHolder(group, &globalWaitTask));
    }
    group.wait();
    assert(globalWaitTask.done());
    if (globalWaitTask.exceptionPtr()) {
      std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
    }
  }

/**
 * Ends the job for the first stream in the schedule list
 */
// The above comment was written by an LLM. 
  void EventProcessor::endJob() {
    // Only on the first stream...
    schedules_[0].endJob();
  }
}  // namespace edm
