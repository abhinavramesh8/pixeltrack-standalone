#ifndef HeterogeneousCore_AlpakaUtilities_allocate_device_h
#define HeterogeneousCore_AlpakaUtilities_allocate_device_h

#include "AlpakaCore/getCachingDeviceAllocator.h"

namespace cms::alpakatools {
  // Allocate device memory
  template <typename TData>
  auto allocate_device(
    const alpaka_common::Extent& extent,
    const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) 
  {
    static const size_t maxAllocationSize = 
      allocator::CachingDeviceAllocator::IntPow(allocator::binGrowth, allocator::maxBin);
    const alpaka_common::Extent nbytes = alpakatools::nbytesFromExtent<TData>(extent);
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      return allocator::getCachingDeviceAllocator().DeviceAllocate<TData>(extent, queue);
    }
    auto buf_ptr {new ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>{
      alpaka::allocBuf<std::byte, alpaka_common::Idx>(alpaka::getDev(queue), nbytes)}};
#if CUDA_VERSION >= 11020
    if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      alpaka::prepareForAsyncCopy(*buf_ptr);
    }
#endif
    return buf_ptr;
  }

  // Free device memory (to be called from unique_ptr)
  inline void free_device(
    void* d_ptr,
    ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>* buf_ptr,
    int device_idx) 
  {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingDeviceAllocator().DeviceFree(d_ptr, device_idx);
    } else {
      delete buf_ptr;
    }
  }
}  // namespace cms::alpakatools

#endif
