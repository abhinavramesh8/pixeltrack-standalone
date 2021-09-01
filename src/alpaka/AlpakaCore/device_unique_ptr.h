#ifndef HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h

#include <memory>

#include "AlpakaCore/allocate_device.h"

namespace cms {
  namespace alpakatools {
    namespace device {
      namespace impl {
        class DeviceDeleter {
        public:
          DeviceDeleter(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>* buffer_ptr) 
            : buf_ptr {buffer_ptr} {}

          void operator()(void* d_ptr) {
            if (d_ptr) {
              cms::alpakatools::free_device(buf_ptr);
            }
          }

        private:
          ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>* buf_ptr;
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<TData, impl::DeviceDeleter>;
    }    // namespace device

    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    typename device::unique_ptr<TData> make_device_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      auto buf_ptr {allocate_device<TData>(extent, queue)};
      void* d_ptr = alpaka::getPtrNative(*buf_ptr);
      return typename device::unique_ptr<TData> {
        reinterpret_cast<TData*>(d_ptr), device::impl::DeviceDeleter {buf_ptr}};
    }

    template <typename TData>
    typename device::unique_ptr<TData> make_device_unique(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      static_assert(std::is_trivially_constructible<TData>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      return make_device_unique_uninitialized<TData>(extent, queue);
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
