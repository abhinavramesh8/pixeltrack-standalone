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
          HostDeleter(alpaka_common::AlpakaHostBuf<std::byte>&& buffer) 
            : buf {std::move(buffer)} {}

          void operator()(void* d_ptr) { 
            if (d_ptr) {
              cms::alpakatools::free_host(d_ptr);
            } 
          }
        
        private:
          alpaka_common::AlpakaHostBuf<std::byte> buf;
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<TData, impl::HostDeleter>;
    }    // namespace host
    
    // Allocate pinned host memory
    template <typename TData>
    typename host::unique_ptr<TData> make_host_unique(
      const alpaka_common::Extent& extent) 
    {
      auto buf {allocate_host<TData>(extent)};
      void* d_ptr = alpaka::getPtrNative(buf);
      return typename host::unique_ptr<TData> {
        reinterpret_cast<TData*>(d_ptr), host::impl::HostDeleter {std::move(buf)}};
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
