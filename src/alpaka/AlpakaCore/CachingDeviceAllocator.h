#ifndef HeterogenousCore_AlpakaUtilities_src_CachingDeviceAllocator_h
#define HeterogenousCore_AlpakaUtilities_src_CachingDeviceAllocator_h

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * Forked to CMSSW by Matti Kortelainen
 */

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#include <cmath>
#include <unordered_set>
#include <memory>
#include <mutex>

#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaCore/deviceAllocatorStatus.h"

/// cms::alpakatools::allocator namespace
namespace cms::alpakatools::allocator {

  /**
 * \addtogroup UtilMgmt
 * @{
 */

  /******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************/

  /**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
  struct CachingDeviceAllocator {
    //---------------------------------------------------------------------
    // Constants
    //---------------------------------------------------------------------

    /// Out-of-bounds bin
    static const unsigned int INVALID_BIN = (unsigned int)-1;

    /// Invalid size
    static const size_t INVALID_SIZE = (size_t)-1;

#ifndef DOXYGEN_SHOULD_SKIP_THIS  // Do not document
    //---------------------------------------------------------------------
    // Type definitions and helper types
    //---------------------------------------------------------------------

    /**
     * Descriptor for device memory allocations
     */
    struct BlockDescriptor {
      void* d_ptr; // Native device pointer
      std::shared_ptr<ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>> buf_ptr; // Device buffer
      size_t bytes; // Size of allocation in bytes
      size_t bytesRequested; // CMS: requested allocation size (for monitoring only)
      unsigned int bin; // Bin enumeration
      int device_idx; // Device
      
      // Constructor (suitable for searching maps for a specific block, given its native device pointer and device id)
      BlockDescriptor(void* ptr, int dev_idx) 
          : d_ptr(ptr),
            buf_ptr(nullptr),
            bytes(0),
            bytesRequested(0),  // CMS
            bin(INVALID_BIN),
            device_idx(dev_idx) {}

      // Constructor (suitable for searching maps for a range of suitable blocks, given a device id)
      BlockDescriptor(int dev_idx)
          : d_ptr(nullptr),
            buf_ptr(nullptr),
            bytes(0),
            bytesRequested(0),  // CMS
            bin(INVALID_BIN),
            device_idx(dev_idx) {}
    };

    struct BlockHashByBytes {
      size_t operator()(const BlockDescriptor& descriptor) const {
        size_t h1 = std::hash<int>{}(descriptor.device_idx);
        size_t h2 = std::hash<size_t>{}(descriptor.bytes);
        return h1 ^ (h2 << 1);
      }
    };

    struct BlockEqualByBytes {
      bool operator()(const BlockDescriptor& a, const BlockDescriptor& b) const {
        return (a.device_idx == b.device_idx && a.bytes == b.bytes);
      }
    };

    struct BlockHashByPtr {
      size_t operator()(const BlockDescriptor& descriptor) const {
        size_t h1 = std::hash<int>{}(descriptor.device_idx);
        size_t h2 = std::hash<void*>{}(descriptor.d_ptr);
        return h1 ^ (h2 << 1);
      }
    };

    struct BlockEqualByPtr {
      bool operator()(const BlockDescriptor& a, const BlockDescriptor& b) const {
        return (a.device_idx == b.device_idx && a.d_ptr == b.d_ptr);
      }
    };

    // CMS: Moved TotalBytes to deviceAllocatorStatus.h

    /// Set type for cached blocks (hashed by size)
    using CachedBlocks = std::unordered_multiset<BlockDescriptor, BlockHashByBytes, BlockEqualByBytes>;

    /// Set type for live blocks (hashed by ptr)
    using BusyBlocks = std::unordered_multiset<BlockDescriptor, BlockHashByPtr, BlockEqualByPtr>;

    // CMS: Moved DeviceCachedBytes to deviceAllocatorStatus.h

    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    /**
     * Integer pow function for unsigned base and exponent
     */
    static unsigned int IntPow(unsigned int base, unsigned int exp) {
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

    /**
     * Round up to the nearest power-of
     */
    void NearestPowerOf(unsigned int &power, size_t &rounded_bytes, unsigned int base, size_t value) {
      power = 0;
      rounded_bytes = 1;

      if (value * base < value) {
        // Overflow
        power = sizeof(size_t) * 8;
        rounded_bytes = size_t(0) - 1;
        return;
      }

      while (rounded_bytes < value) {
        rounded_bytes *= base;
        power++;
      }
    }

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    // CMS: use std::mutex instead of cub::Mutex, declare mutable
    mutable std::mutex mutex;  /// Mutex for thread-safety

    unsigned int bin_growth;  /// Geometric growth factor for bin-sizes
    unsigned int min_bin;     /// Minimum bin enumeration
    unsigned int max_bin;     /// Maximum bin enumeration

    size_t min_bin_bytes;     /// Minimum bin size
    size_t max_bin_bytes;     /// Maximum bin size
    size_t max_cached_bytes;  /// Maximum aggregate cached bytes per device

    bool debug;        /// Whether or not to print (de)allocation events to stdout

    DeviceCachedBytes cached_bytes;  /// Map of device to aggregate cached bytes on that device
    CachedBlocks cached_blocks;   /// Set of cached device allocations available for reuse
    BusyBlocks live_blocks;       /// Set of live device allocations currently in use

#endif  // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingDeviceAllocator(
        unsigned int bin_growth,                 ///< Geometric growth factor for bin-sizes
        unsigned int min_bin = 1,                ///< Minimum bin (default is bin_growth ^ 1)
        unsigned int max_bin = INVALID_BIN,      ///< Maximum bin (default is no max bin)
        size_t max_cached_bytes = INVALID_SIZE,  ///< Maximum aggregate cached bytes per device (default is no limit)
        bool debug = false)  ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
        : bin_growth(bin_growth),
          min_bin(min_bin),
          max_bin(max_bin),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes(max_cached_bytes),
          debug(debug) {}

    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth          = 8
     * - \p min_bin             = 3
     * - \p max_bin             = 7
     * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes per device
     */
    CachingDeviceAllocator(/*bool skip_cleanup = false, */bool debug = false)
        : bin_growth(8),
          min_bin(3),
          max_bin(7),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes((max_bin_bytes * 3) - 1),
          debug(debug) {}

    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     *
     * Changing the ceiling of cached bytes does not cause any allocations (in-use or
     * cached-in-reserve) to be freed.  See \p FreeAllCached().
     */
    void SetMaxCachedBytes(size_t max_cached_bytes) {
      // Lock
      mutex.lock();

      if (debug)
        // CMS: use raw printf
        printf("Changing max_cached_bytes (%lld -> %lld)\n",
               (long long)this->max_cached_bytes,
               (long long)max_cached_bytes);

      this->max_cached_bytes = max_cached_bytes;

      mutex.unlock();
    }

    /**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    template <typename TData>
    auto DeviceAllocate(
        const alpaka_common::Extent& extent,                     ///< [in] Extent of the allocation
        const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device) ///< [in] The device to be associated with this allocation
    {
      auto device_idx = getIdxOfDev(device);
      size_t bytes = alpakatools::nbytesFromExtent<TData>(extent);
      
      // Create a block descriptor for the requested allocation
      bool found = false;
      BlockDescriptor search_key(device_idx);
      search_key.bytesRequested = bytes;  // CMS
      NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

      if (search_key.bin > max_bin) {
        // Bin is greater than our maximum bin: allocate the request
        // exactly and give out-of-bounds bin.  It will not be cached
        // for reuse when returned.
        search_key.bin = INVALID_BIN;
        search_key.bytes = bytes;
      } else {
        // Search for a suitable cached allocation: lock
        mutex.lock();

        if (search_key.bin < min_bin) {
          // Bin is less than minimum bin: round up
          search_key.bin = min_bin;
          search_key.bytes = min_bin_bytes;
        }

        // Find a cached block on the same device in the same bin
        auto block_itr = cached_blocks.find(search_key);
        if (block_itr != cached_blocks.end()) {
          // Reuse existing cache block.  Insert into live blocks.
          found = true;
          search_key = *block_itr;
          live_blocks.insert(search_key);

          // Remove from free blocks
          cached_bytes[device_idx].free -= search_key.bytes;
          cached_bytes[device_idx].live += search_key.bytes;
          cached_bytes[device_idx].liveRequested += search_key.bytesRequested; // CMS 

          /*if (debug)
            // CMS: improved debug message
            // CMS: use raw printf
            printf(
                "\tDevice %d reused cached block at %p (%lld bytes) for stream %lld, event %lld (previously "
                "associated with stream %lld, event %lld).\n",
                device,
                search_key.d_ptr,
                (long long)search_key.bytes,
                (long long)search_key.associated_stream,
                (long long)search_key.ready_event,
                (long long)block_itr->associated_stream,
                (long long)block_itr->ready_event);*/

          cached_blocks.erase(block_itr);
        }
        // Done searching: unlock
        mutex.unlock();
      }

      // Allocate the block if necessary
      if (!found) {
        auto buf {alpaka::allocBuf<std::byte, alpaka_common::Idx>(
          device, static_cast<alpaka_common::Extent>(search_key.bytes))};
        search_key.d_ptr = alpaka::getPtrNative(buf);
        search_key.buf_ptr = std::make_shared<ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<std::byte>>(
          std::move(buf)
        );

        // Insert into live blocks
        mutex.lock();
        live_blocks.insert(search_key);
        cached_bytes[device_idx].live += search_key.bytes;
        cached_bytes[device_idx].liveRequested += search_key.bytesRequested;  // CMS
        mutex.unlock();
        
        /*if (debug)
          // CMS: improved debug message
          // CMS: use raw printf
          printf("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld, event %lld).\n",
                 device,
                 search_key.d_ptr,
                 (long long)search_key.bytes,
                 (long long)search_key.associated_stream,
                 (long long)search_key.ready_event);*/
      }

      /*if (debug)
        // CMS: use raw printf
        printf("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
               (long long)cached_blocks.size(),
               (long long)cached_bytes[device].free,
               (long long)live_blocks.size(),
               (long long)cached_bytes[device].live);*/
      return search_key.buf_ptr.get();  
    }

    /**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
     */
    void DeviceFree(void* d_ptr, int device_idx) 
    {
      // Lock
      mutex.lock();

      // Find corresponding block descriptor
      BlockDescriptor search_key(d_ptr, device_idx);
      auto block_itr = live_blocks.find(search_key);
      if (block_itr != live_blocks.end()) {
        // Remove from live blocks
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_bytes[device_idx].live -= search_key.bytes;
        cached_bytes[device_idx].liveRequested -= search_key.bytesRequested;  // CMS

        // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
        if ((search_key.bin != INVALID_BIN) && (cached_bytes[device_idx].free + search_key.bytes <= max_cached_bytes)) {
          // Insert returned allocation into free blocks
          cached_blocks.insert(search_key);
          cached_bytes[device_idx].free += search_key.bytes;

          /*if (debug)
            // CMS: improved debug message
            // CMS: use raw printf
            printf(
                "\tDevice %d returned %lld bytes at %p from associated stream %lld, event %lld.\n\t\t %lld available "
                "blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                device,
                (long long)search_key.bytes,
                d_ptr,
                (long long)search_key.associated_stream,
                (long long)search_key.ready_event,
                (long long)cached_blocks.size(),
                (long long)cached_bytes[device].free,
                (long long)live_blocks.size(),
                (long long)cached_bytes[device].live);*/
        }
      }

      // Unlock
      mutex.unlock();

      /*if (!recached and debug) {
        // CMS: improved debug message
        printf(
            "\tDevice %d freed %lld bytes at %p from associated stream %lld, event %lld.\n\t\t  %lld available "
            "blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
            device,
            (long long)search_key.bytes,
            d_ptr,
            (long long)search_key.associated_stream,
            (long long)search_key.ready_event,
            (long long)cached_blocks.size(),
            (long long)cached_bytes[device].free,
            (long long)live_blocks.size(),
            (long long)cached_bytes[device].live);
      }*/
    }

    // CMS: give access to cache allocation status
    DeviceCachedBytes CacheStatus() const {
      std::unique_lock mutex_locker(mutex);
      return cached_bytes;
    }
  };

  /** @} */  // end group UtilMgmt

}  // namespace cms::alpakatools::allocator

#endif
