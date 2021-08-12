#ifndef HeterogeneousCore_AlpakaCore_src_getCachingHostAllocator
#define HeterogeneousCore_AlpakaCore_src_getCachingHostAllocator

#include <iomanip>
#include <iostream>

#include "CachingHostAllocator.h"

// #include "getCachingDeviceAllocator.h"

namespace cms::alpaka::allocator {
  // TODO : move this to CachingDeviceAllocator.h later
  /**
   * Integer pow function for unsigned base and exponent
   */
  unsigned int IntPow(unsigned int base, unsigned int exp) {
    unsigned int retval = 1;
    while (exp > 0) {
      if (exp & 1) {
        retval = retval * base;  // multiply the result by the current base
      }
      base = base * base;  // square the base
      exp = exp >> 1;      // divide the exponent in half
    }
    return retval;
  }

  // TODO : move all the stuff below to getCachingDeviceAllocator.h later
  constexpr unsigned int binGrowth = 2;
  constexpr unsigned int minBin = 8;
  constexpr unsigned int maxBin = 30;
  constexpr size_t maxCachedBytes = 0;
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;
  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    const size_t numberOfDevices {::alpaka::getDevCount<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>()};
    for (size_t i = 0; i < numberOfDevices; ++i) {
      const auto device {::alpaka::getDevByIdx<ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1>(i)};
      const size_t freeMemory {::alpaka::getFreeMemBytes(device)};
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  template <class TData>
  inline CachingHostAllocator<TData>& getCachingHostAllocator() {
    if (debug) {
      std::cout << "CachingHostAllocator settings\n"
                << "  bin growth " << binGrowth << "\n"
                << "  min bin    " << minBin << "\n"
                << "  max bin    " << maxBin << "\n"
                << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = IntPow(binGrowth, bin);
        if (binSize >= (1 << 30) and binSize % (1 << 30) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 30) << " GB\n";
        } else if (binSize >= (1 << 20) and binSize % (1 << 20) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 20) << " MB\n";
        } else if (binSize >= (1 << 10) and binSize % (1 << 10) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 10) << " kB\n";
        } else {
          std::cout << "    " << std::setw(9) << binSize << " B\n";
        }
      }
      std::cout << "  maximum amount of cached memory: " << (minCachedBytes() >> 20) << " MB\n";
    }

    // the public interface is thread safe
    static CachingHostAllocator<TData> allocator{binGrowth,
                                                 minBin,
                                                 maxBin,
                                                 minCachedBytes(),
                                                 false,  // do not skip cleanup
                                                 debug};
    return allocator;
  }
}  // namespace cms::alpaka::allocator

#endif