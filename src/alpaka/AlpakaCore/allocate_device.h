#ifndef HeterogeneousCore_AlpakaUtilities_allocate_device_h
#define HeterogeneousCore_AlpakaUtilities_allocate_device_h

#include "AlpakaCore/getCachingDeviceAllocator.h"

namespace cms::alpaka {
  // Allocate device memory
  template <typename TData>
  auto allocate_device(
    const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device,
    const alpaka_common::Extent& extent,
    const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
  {
    static const size_t maxAllocationSize = 
      allocator::CachingDeviceAllocator<TData>::IntPow(allocator::binGrowth, allocator::maxBin);
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      const size_t nbytes = alpakatools::nbytesFromExtent<TData>(extent);
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      return allocator::getCachingDeviceAllocator<TData>().DeviceAllocate(device, extent, queue);
    }
    auto buf {::alpaka::allocBuf<TData, alpaka_common::Idx>(device, extent)};
#if CUDA_VERSION >= 11020
    if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ::alpaka::prepareForAsyncCopy(buf);
    }
#endif
    return buf;
  }

  // Free device memory (to be called from unique_ptr)
  template <typename TData>
  void free_device(
    const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 &device,
    ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> &buf) 
  {
    auto dev_buf {std::move(buf)};
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingDeviceAllocator<TData>().DeviceFree(device, dev_buf);
    }
  }
}  // namespace cms::alpaka

#endif
