#ifndef AlpakaDataFormats_Common_ProductBase_h
#define AlpakaDataFormats_Common_ProductBase_h

#include <atomic>
#include <memory>

#include "AlpakaCore/SharedEventPtr.h"
#include "AlpakaCore/SharedStreamPtr.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  namespace impl {
    class ScopedContextBase;
  }

  /**
     * Base class for all instantiations of CUDA<T> to hold the
     * non-T-dependent members.
     */
  class ProductBase {
  public:
    ProductBase() = default;  // Needed only for ROOT dictionary generation
    ~ProductBase();

    ProductBase(const ProductBase&) = delete;
    ProductBase& operator=(const ProductBase&) = delete;
    ProductBase(ProductBase&& other)
        : stream_{std::move(other.stream_)},
          event_{std::move(other.event_)},
          mayReuseStream_{other.mayReuseStream_.load()},
          device_{other.device_} {}
    ProductBase& operator=(ProductBase&& other) {
      stream_ = std::move(other.stream_);
      event_ = std::move(other.event_);
      mayReuseStream_ = other.mayReuseStream_.load();
      device_ = other.device_;
      return *this;
    }

    bool isValid() const { return stream_.get() != nullptr; }
    bool isAvailable() const;

    int device() const { return device_; }

    // cudaStream_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ::cms::alpakatools::ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    ::ALPAKA_ACCELERATOR_NAMESPACE::Queue& stream() const { return *(stream_.get()); }

    // cudaEvent_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the ::cms::alpakatools::ScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    alpaka::Event<::ALPAKA_ACCELERATOR_NAMESPACE::Queue>& event() const { return *(event_.get()); }

  protected:
    explicit ProductBase(int device, SharedStreamPtr stream, SharedEventPtr event)
        : stream_{std::move(stream)}, event_{std::move(event)}, device_{device} {}

  private:
    friend class impl::ScopedContextBase;
    friend class ScopedContextProduce;

    // The following function is intended to be used only from ScopedContext
    const SharedStreamPtr& streamPtr() const { return stream_; }

    bool mayReuseStream() const {
      bool expected = true;
      bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
      // If the current thread is the one flipping the flag, it may
      // reuse the stream.
      return changed;
    }

    // The cudaStream_t is really shared among edm::Event products, so
    // using shared_ptr also here
    SharedStreamPtr stream_;  //!
    // shared_ptr because of caching in ::cms::alpakatools::EventCache
    SharedEventPtr event_;  //!

    // This flag tells whether the CUDA stream may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same stream.
    mutable std::atomic<bool> mayReuseStream_ = true;  //!

    // The CUDA device associated with this product
    int device_ = -1;  //!
  };

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_Common_ProductBase_h
