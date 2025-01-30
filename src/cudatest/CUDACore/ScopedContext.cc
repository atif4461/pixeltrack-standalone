#include "CUDACore/ScopedContext.h"

#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"

#include "chooseDevice.h"

namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };

/**
 * @brief Callback function executed after CUDA operation completion
 * @param streamId Identifier of the CUDA stream that triggered this callback
 * @param status Resulting status of the CUDA operation
 * @param data Pointer to user-provided data associated with the callback
 */
// The above comment was written by an LLM. 
  void CUDART_CB cudaScopedContextCallback(cudaStream_t streamId, cudaError_t status, void* data) {
    std::unique_ptr<CallbackData> guard{reinterpret_cast<CallbackData*>(data)};
    edm::WaitingTaskWithArenaHolder& waitingTaskHolder = guard->holder;
    int device = guard->device;
    if (status == cudaSuccess) {
      //std::cout << " GPU kernel finished (in callback) device " << device << " CUDA stream "
      //          << streamId << std::endl;
      waitingTaskHolder.doneWaiting(nullptr);
    } else {
      // wrap the exception in a try-catch block to let GDB "catch throw" break on it
      try {
        auto error = cudaGetErrorName(status);
        auto message = cudaGetErrorString(status);
        throw std::runtime_error("Callback of CUDA stream " +
                                 std::to_string(reinterpret_cast<unsigned long>(streamId)) + " in device " +
                                 std::to_string(device) + " error " + std::string(error) + ": " + std::string(message));
      } catch (std::exception&) {
        waitingTaskHolder.doneWaiting(std::current_exception());
      }
    }
  }
}  // namespace

namespace cms::cuda {
  namespace impl {
/**
 * Constructor initializing scoped context base with specified stream ID
 * @param streamID EDM stream identifier
 */
// The above comment was written by an LLM. 
    ScopedContextBase::ScopedContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
      cudaCheck(cudaSetDevice(currentDevice_));
      stream_ = getStreamCache().get();
    }

/**
 * Constructs an object initializing device and stream
 * @param data reference to product base containing device and stream information
 */
// The above comment was written by an LLM. 
    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      cudaCheck(cudaSetDevice(currentDevice_));
      if (data.mayReuseStream()) {
        stream_ = data.streamPtr();
      } else {
        stream_ = getStreamCache().get();
      }
    }

/**
 * Constructs a scoped context base object with specified device and shared stream
 * @param device The device identifier
 * @param stream The shared stream pointer
 */
// The above comment was written by an LLM. 
    ScopedContextBase::ScopedContextBase(int device, SharedStreamPtr stream)
        : currentDevice_(device), stream_(std::move(stream)) {
      cudaCheck(cudaSetDevice(currentDevice_));
    }

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(int dataDevice,
                                                     cudaStream_t dataStream,
                                                     bool available,
                                                     cudaEvent_t dataEvent) {
      if (dataDevice != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw std::runtime_error("Handling data from multiple devices is not yet supported");
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          cudaCheck(cudaStreamWaitEvent(stream(), dataEvent, 0), "Failed to make a stream to wait for an event");
        }
      }
    }

/**
 * Enqueues a callback on a CUDA stream to execute when all preceding commands have completed
 * @param device the identifier of the device associated with the callback
 * @param stream the CUDA stream on which to enqueue the callback
 */
// The above comment was written by an LLM. 
    void ScopedContextHolderHelper::enqueueCallback(int device, cudaStream_t stream) {
      cudaCheck(
          cudaStreamAddCallback(stream, cudaScopedContextCallback, new CallbackData{waitingTaskHolder_, device}, 0));
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() {
    holderHelper_.enqueueCallback(device(), stream());
    if (contextState_) {
      contextState_->set(device(), streamPtr());
    }
  }

/**
 * Throws an exception when insertNextTask is called without proper state initialization
 */
// The above comment was written by an LLM. 
  void ScopedContextAcquire::throwNoState() {
    throw std::runtime_error(
        "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
        "ContextState, but that was not the case");
  }

  ////////////////////

  ScopedContextProduce::~ScopedContextProduce() {
    // Intentionally not checking the return value to avoid throwing
    // exceptions. If this call would fail, we should get failures
    // elsewhere as well.
    cudaEventRecord(event_.get(), stream());
  }

  ////////////////////

  ScopedContextTask::~ScopedContextTask() { holderHelper_.enqueueCallback(device(), stream()); }
}  // namespace cms::cuda
