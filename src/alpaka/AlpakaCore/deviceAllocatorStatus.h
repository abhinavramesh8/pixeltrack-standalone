#ifndef HeterogeneousCore_AlpakaUtilities_deviceAllocatorStatus_h
#define HeterogeneousCore_AlpakaUtilities_deviceAllocatorStatus_h

#include <map>

#include "AlpakaCore/alpakaConfig.h"

namespace cms {
  namespace alpakatools {
    namespace allocator {
      struct TotalBytes {
        size_t free;
        size_t live;
        size_t liveRequested;  // CMS: monitor also requested amount
        TotalBytes() { free = live = liveRequested = 0; }
      };

      inline const auto& getDevs() {
        static const auto devices = alpaka::getDevs<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>();
        return devices;
      }
      
      inline int getIdxOfDev(const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device) {
        return (std::find(getDevs().begin(), getDevs().end(), device) - 
                getDevs().begin());
      }
      
      // Map device to the number of bytes cached by it
      using DeviceCachedBytes = std::map<int, TotalBytes>;
    }  // namespace allocator
  }  // namespace alpakatools
}  // namespace cms

#endif
