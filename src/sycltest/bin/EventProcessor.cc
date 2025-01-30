#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
/**
 * Constructor initializing event processing with specified parameters
 * @param warmupEvents number of events for warm up phase
 * @param maxEvents maximum number of events to process
 * @param runForMinutes duration in minutes for which to run
 * @param numberOfStreams number of streams for parallel processing
 * @param path vector of strings representing paths
 * @param esproducers vector of strings representing es producers
 * @param datadir file system path for data directory
 * @param validation boolean flag indicating validation mode
 */
// The above comment was written by an LLM. 
  EventProcessor::EventProcessor(int warmupEvents,
                                 int maxEvents,
                                 int runForMinutes,
                                 int numberOfStreams,
                                 std::vector<std::string> const& path,
                                 std::vector<std::string> const& esproducers,
                                 std::filesystem::path const& datadir,
                                 bool validation)
      : source_(maxEvents, runForMinutes, registry_, datadir, validation),
        warmupEvents_(warmupEvents),
        maxEvents_(source_.maxEvents()),
        runForMinutes_(runForMinutes) {
    for (auto const& name : esproducers) {
      pluginManager_.load(name);
      auto esp = ESPluginFactory::create(name, datadir);
      esp->produce(eventSetup_);
    }

    schedules_.reserve(numberOfStreams);
    for (int i = 0; i < numberOfStreams; ++i) {
      schedules_.emplace_back(registry_, pluginManager_, &source_, &eventSetup_, i, path);
    }
  }

/**
 * Initializes the event processor with initial events to achieve stable state
 */
// The above comment was written by an LLM. 
  void EventProcessor::warmUp() {
    if (warmupEvents_ <= 0)
      return;

    // Configure the source for the warmup step
    source_.reconfigure(warmupEvents_, -1);
    process();
  }

/**
 * Runs the event processing to completion reconfiguring the source 
 * with specified maximum events and runtime in minutes prior 
 * to initiating the processing sequence */
// The above comment was written by an LLM. 
  void EventProcessor::runToCompletion() {
    // Configure the source for the actual reconstrction
    source_.reconfigure(maxEvents_, runForMinutes_);
    process();
  }

/**
 * Starts event processing by initiating tasks asynchronously 
 * and waiting for their completion while handling exceptions */
// The above comment was written by an LLM. 
  void EventProcessor::process() {
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
