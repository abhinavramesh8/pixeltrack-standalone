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
      
      inline size_t getIdxOfDev(const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device) {
        return static_cast<size_t>(
          std::find(getDevs().begin(), getDevs().end(), device) - 
          getDevs().begin()
        );
      }

      struct DeviceIdxCompare {
        bool operator()(
          const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& a, 
          const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& b) const {
          return (getIdxOfDev(a) < getIdxOfDev(b));  
        }
      };
      
      /// Map device to the number of bytes cached by it
      using DeviceCachedBytes = std::map<ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1, TotalBytes, DeviceIdxCompare>;
    }  // namespace allocator
  }  // namespace alpakatools
}  // namespace cms

#endif
