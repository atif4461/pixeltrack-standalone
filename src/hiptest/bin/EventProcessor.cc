#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
/**
 * Constructor initializing event processor with specified parameters.
 * @param warmupEvents Number of events to process during warm up phase.
 * @param maxEvents Maximum number of events to process.
 * @param runForMinutes Duration in minutes to run event processing.
 * @param numberOfStreams Number of streams for parallel event processing.
 * @param path Vector of strings representing paths for event processing.
 * @param esproducers Vector of strings representing event setup producers.
 * @param datadir File system path for data directory.
 * @param validation Flag indicating whether validation is enabled.
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
 * Initializes the event processing system with a specified number of events 
 * to prepare it for subsequent processing steps
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
 * and waiting for their completion while handling exceptions.
 */
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
 * Ends the job processing for the first stream
 */
// The above comment was written by an LLM. 
  void EventProcessor::endJob() {
    // Only on the first stream...
    schedules_[0].endJob();
  }
}  // namespace edm
