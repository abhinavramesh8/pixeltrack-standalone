#ifndef HeterogeneousCore_AlpakaUtilities_allocate_host_h
#define HeterogeneousCore_AlpakaUtilities_allocate_host_h

#include "AlpakaCore/getCachingHostAllocator.h"

namespace cms::alpaka {
  // Allocate pinned host memory (to be called from unique_ptr)
  template <typename TData>
  auto allocate_host(
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
      return allocator::getCachingHostAllocator<TData>().HostAllocate(extent, queue);
    }
    return alpakatools::allocHostBuf<TData>(extent);
  }

  // Free pinned host memory (to be called from unique_ptr)
  template <typename TData>
  void free_host(alpaka_common::AlpakaHostBuf<TData> &buf) {
    auto host_buf {std::move(buf)};
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingHostAllocator<TData>().HostFree(host_buf);
    }
  }
}  // namespace cms::alpaka

#endif
