#ifndef HeterogeneousCore_AlpakaCore_ContextState_h
#define HeterogeneousCore_AlpakaCore_ContextState_h

#include <memory>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/SharedStreamPtr.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  /**
     * The purpose of this class is to deliver the device and CUDA stream
     * information from ExternalWork's acquire() to producer() via a
     * member/StreamCache variable.
     */
  class ContextState {
  public:
    ContextState() = default;
    ~ContextState() = default;

    ContextState(const ContextState&) = delete;
    ContextState& operator=(const ContextState&) = delete;
    ContextState(ContextState&&) = delete;
    ContextState& operator=(ContextState&& other) = delete;

  private:
    friend class ScopedContextAcquire;
    friend class ScopedContextProduce;
    friend class ScopedContextTask;

    void set(int device, SharedStreamPtr stream) {
      throwIfStream();
      device_ = device;
      stream_ = std::move(stream);
    }

    int device() const { return device_; }

    const SharedStreamPtr& streamPtr() const {
      throwIfNoStream();
      return stream_;
    }

    SharedStreamPtr releaseStreamPtr() {
      throwIfNoStream();
      // This function needs to effectively reset stream_ (i.e. stream_
      // must be empty after this function). This behavior ensures that
      // the SharedStreamPtr is not hold for inadvertedly long (i.e. to
      // the next event), and is checked at run time.
      return std::move(stream_);
    }

    void throwIfStream() const;
    void throwIfNoStream() const;

    SharedStreamPtr stream_;
    int device_;
  };

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_ContextState_h
