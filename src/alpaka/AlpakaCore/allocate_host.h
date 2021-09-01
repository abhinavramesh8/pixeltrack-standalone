#ifndef HeterogeneousCore_AlpakaUtilities_allocate_host_h
#define HeterogeneousCore_AlpakaUtilities_allocate_host_h

#include "AlpakaCore/getCachingHostAllocator.h"

namespace cms::alpakatools {
  // Allocate pinned host memory (to be called from unique_ptr)
  template <typename TData>
  auto allocate_host(
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
      return allocator::getCachingHostAllocator().HostAllocate<TData>(extent, queue);
    }
    auto buf_ptr {new alpaka_common::AlpakaHostBuf<std::byte>{
      alpakatools::allocHostBuf<std::byte>(nbytes)}};
    if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      alpaka::prepareForAsyncCopy(*buf_ptr);
    }
    return buf_ptr;
  }

  // Free pinned host memory (to be called from unique_ptr)
  inline void free_host(alpaka_common::AlpakaHostBuf<std::byte>* buf_ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingHostAllocator().HostFree(buf_ptr);
    } else {
      delete buf_ptr;
    }
  }
}  // namespace cms::alpakatools

#endif
