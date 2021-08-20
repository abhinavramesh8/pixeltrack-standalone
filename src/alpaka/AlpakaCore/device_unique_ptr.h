#ifndef HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h

#include <memory>

#include "AlpakaCore/allocate_device.h"

namespace cms {
  namespace alpakatools {
    namespace device {
      namespace impl {
        template <typename TData>
        class DeviceDeleter {
        public:
          DeviceDeleter(const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& dev) : device_{dev} {}

          void operator()(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> *buf_ptr) {
            cms::alpakatools::free_device<TData>(device_, *buf_ptr);
            delete buf_ptr;
          }

        private:
          ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device_;
        };
      }  // namespace impl

      template <typename TData>
      using unique_ptr = std::unique_ptr<ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData>, impl::DeviceDeleter<TData> >;
    }    // namespace device

    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    typename device::unique_ptr<TData> make_device_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      const auto& dev {ALPAKA_ACCELERATOR_NAMESPACE::device};
      auto buf_ptr {new ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> {
        allocate_device<TData>(dev, extent, queue)}};
      return typename device::unique_ptr<TData> {
        buf_ptr, device::impl::DeviceDeleter<TData> {dev}};
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
