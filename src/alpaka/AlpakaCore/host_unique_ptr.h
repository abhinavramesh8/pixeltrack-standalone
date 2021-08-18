#ifndef HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h

#include <memory>

#include "AlpakaCore/allocate_host.h"

namespace cms {
  namespace alpaka {
    namespace host {
      namespace impl {
        template <typename TData>
        class HostDeleter {
        public:
          void operator()(alpaka_common::AlpakaHostBuf<TData> *buf_ptr) { 
            cms::alpaka::free_host<TData>(*buf_ptr);
            delete buf_ptr; 
          }
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<alpaka_common::AlpakaHostBuf<TData>, impl::HostDeleter<TData> >;
    }    // namespace host

    // Allocate pinned host memory
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      static_assert(std::is_trivially_constructible<TData>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      auto buf_ptr {new AlpakaHostBuf<TData> {allocate_host<TData>(extent, queue)}};
      return typename host::unique_ptr<TData> {buf_ptr};
    }

    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      auto buf_ptr {new AlpakaHostBuf<TData> {allocate_host<TData>(extent, queue)}};
      return typename host::unique_ptr<TData> {buf_ptr};
    }
  }  // namespace alpaka
}  // namespace cms

#endif
