#ifndef HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h

#include <memory>
#include <type_traits>

#include "AlpakaCore/allocate_device.h"
#include "AlpakaCore/host_unique_ptr.h"

namespace cms {
  namespace alpakatools {
    namespace device {
      namespace impl {
        class DeviceDeleter {
        public:
          DeviceDeleter(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>* buffer_ptr, int dev_idx) 
            : buf_ptr {buffer_ptr}, device_idx {dev_idx} {}

          void operator()(void* d_ptr) {
            if (d_ptr) {
              cms::alpakatools::free_device(d_ptr, buf_ptr, device_idx);
            }
          }

        private:
          ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>* buf_ptr;
          int device_idx;
        };
      }  // namespace impl
      template <typename TData>
      using unique_ptr = 
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        std::unique_ptr<TData, impl::DeviceDeleter>;
#else
        host::unique_ptr<TData>;
#endif
    }    // namespace device

    // No check for the trivial constructor, make it clear in the interface
    template <typename TData>
    auto make_device_unique_uninitialized(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto buf_ptr {allocate_device<TData>(extent, queue)};
      auto device_idx {allocator::getIdxOfDev(alpaka::getDev(*buf_ptr))};
      void* d_ptr = alpaka::getPtrNative(*buf_ptr);
      return typename device::unique_ptr<TData> {
        reinterpret_cast<TData*>(d_ptr), device::impl::DeviceDeleter {buf_ptr, device_idx}};
#else
      return make_host_unique_uninitialized<TData>(extent, queue);
#endif
    }

    template <typename TData>
    auto make_device_unique(
      const alpaka_common::Extent& extent, 
      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
    {
      /*static_assert(std::is_trivially_constructible<TData>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");*/
      return make_device_unique_uninitialized<TData>(extent, queue);
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
