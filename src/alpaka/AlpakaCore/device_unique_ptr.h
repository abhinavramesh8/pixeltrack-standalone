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

    template <typename TData>
    auto make_device_unique(const alpaka_common::Extent& extent) 
    {
      const auto& device = ALPAKA_ACCELERATOR_NAMESPACE::device;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto buf_ptr {allocate_device<TData>(extent, device)};
      auto device_idx {allocator::getIdxOfDev(device)};
      void* d_ptr = alpaka::getPtrNative(*buf_ptr);
      return typename device::unique_ptr<TData> {
        reinterpret_cast<TData*>(d_ptr), device::impl::DeviceDeleter {buf_ptr, device_idx}};
#else
      return make_host_unique<TData>(extent, ALPAKA_ACCELERATOR_NAMESPACE::Queue{device});
#endif
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
