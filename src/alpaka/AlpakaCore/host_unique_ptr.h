#ifndef HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_host_unique_ptr_h

#include <memory>

#include "AlpakaCore/allocate_host.h"

namespace cms {
  namespace alpakatools {
    namespace host {
      namespace impl {

        class HostDeleter {
        public:
          HostDeleter(alpaka_common::AlpakaHostBuf<std::byte>* buffer_ptr) 
            : buf_ptr {buffer_ptr} {}

          void operator()(void* d_ptr) { 
            if (d_ptr) {
              cms::alpakatools::free_host(d_ptr, buf_ptr);
            } 
          }
        
        private:
          alpaka_common::AlpakaHostBuf<std::byte>* buf_ptr;
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<TData, impl::HostDeleter>;
    }    // namespace host
    
    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      auto buf_ptr {allocate_host<TData>(extent, queue)};
      void* d_ptr = alpaka::getPtrNative(*buf_ptr);
      return typename host::unique_ptr<TData> {
        reinterpret_cast<TData*>(d_ptr), host::impl::HostDeleter {buf_ptr}};
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
