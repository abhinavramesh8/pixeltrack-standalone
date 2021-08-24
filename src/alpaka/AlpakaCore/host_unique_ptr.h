#ifndef HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h

#include <memory>

#include "AlpakaCore/allocate_host.h"

namespace cms {
  namespace alpakatools {
    namespace host {
      namespace impl {
        template <typename TData>
        class HostDeleter {
        public:
          void operator()(alpaka_common::AlpakaHostBuf<TData> *buf_ptr) { 
            if (buf_ptr) {
              cms::alpakatools::free_host<TData>(*buf_ptr);
              delete buf_ptr;
            } 
          }
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<alpaka_common::AlpakaHostBuf<TData>, impl::HostDeleter<TData> >;
    }    // namespace host
    
    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      auto buf_ptr {new alpaka_common::AlpakaHostBuf<TData> {allocate_host<TData>(extent, queue)}};
      return typename host::unique_ptr<TData> {buf_ptr};
    }

    // Allocate pinned host memory
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      static_assert(std::is_trivially_constructible<TData>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      return make_host_unique_uninitialized<TData>(extent, queue);
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
