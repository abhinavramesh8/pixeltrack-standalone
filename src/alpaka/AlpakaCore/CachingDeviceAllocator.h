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
#include <map>
#include <set>
#include <mutex>

#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaCore/deviceAllocatorStatus.h"

/// cms::alpaka::allocator namespace
namespace cms::alpaka::allocator {

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
 * The allocator is thread-safe and queue-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_queue.  Once freed,
 *   the allocation becomes available immediately for reuse within the \p active_queue
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other queues when all prior work submitted to \p active_queue has completed.
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
  template <typename TData>
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
      ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> buf;         // Device buffer
      size_t bytes;                                                     // Size of allocation in bytes
      size_t bytesRequested;                                            // CMS: requested allocatoin size (for monitoring only)
      unsigned int bin;                                                 // Bin enumeration
      ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1 device;                     // Device
      ALPAKA_ACCELERATOR_NAMESPACE::Queue associated_queue;             // Associated associated_queue
      ::alpaka::Event<ALPAKA_ACCELERATOR_NAMESPACE::Queue> ready_event; // Signal when associated queue has run to the point at which this block was freed

      // Constructor (suitable for searching maps for a specific block, given its buffer and device)
      BlockDescriptor(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> buffer, 
                      const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& dev)
          : buf(std::move(buffer)),
            bytes(0),
            bytesRequested(0),  // CMS
            bin(INVALID_BIN),
            device(dev),
            associated_queue(device),
            ready_event(device) {}

      // Constructor (suitable for searching maps for a range of suitable blocks, given a device)
      BlockDescriptor(const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& dev,
                      const ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue)
          : buf(cms::alpakatools::allocDeviceBuf<TData>(alpaka_common::Extent{})),
            bytes(0),
            bytesRequested(0),  // CMS
            bin(INVALID_BIN),
            device(dev),
            associated_queue(queue),
            ready_event(device) {}

      // Comparison functor for comparing devices
      static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b) {
        if (a.device == b.device)
          return (::alpaka::getPtrNative(a.buf) < ::alpaka::getPtrNative(b.buf));
        else
          return DeviceIdxCompare{}(a.device, b.device);
      }

      // Comparison functor for comparing allocation sizes
      static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b) {
        if (a.device == b.device)
          return (a.bytes < b.bytes);
        else
          return DeviceIdxCompare{}(a.device, b.device);
      }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    // CMS: Moved TotalBytes to deviceAllocatorStatus.h

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

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

    const bool
        skip_cleanup;  /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The backend runtime may have already shut down for statically declared allocators)
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
        bool skip_cleanup =
            false,  ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
        bool debug = false)  ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
        : bin_growth(bin_growth),
          min_bin(min_bin),
          max_bin(max_bin),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes(max_cached_bytes),
          skip_cleanup(skip_cleanup),
          debug(debug),
          cached_blocks(BlockDescriptor::SizeCompare),
          live_blocks(BlockDescriptor::PtrCompare) {}

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
    CachingDeviceAllocator(bool skip_cleanup = false, bool debug = false)
        : bin_growth(8),
          min_bin(3),
          max_bin(7),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes((max_bin_bytes * 3) - 1),
          skip_cleanup(skip_cleanup),
          debug(debug),
          cached_blocks(BlockDescriptor::SizeCompare),
          live_blocks(BlockDescriptor::PtrCompare) {}

    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     *
     * Changing the ceiling of cached bytes does not cause any allocations (in-use or
     * cached-in-reserve) to be freed.  See \p FreeAllCached().
     */
    void SetMaxCachedBytes(size_t max_cached_bytes) {
      // Lock
      // CMS: use RAII instead of (un)locking explicitly
      std::unique_lock mutex_locker(mutex);

      if (debug)
        // CMS: use raw printf
        printf("Changing max_cached_bytes (%lld -> %lld)\n",
               (long long)this->max_cached_bytes,
               (long long)max_cached_bytes);

      this->max_cached_bytes = max_cached_bytes;

      // Unlock (redundant, kept for style uniformity)
      mutex_locker.unlock();
    }

    /**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_queue
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * queues when all prior work submitted to \p active_queue has completed.
     */
    auto DeviceAllocate(
        const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device,     ///< [in] Device on which to place the allocation
        const Extent& extent,                                    ///< [in] Extent of the allocation
        const ALPAKA_ACCELERATOR_NAMESPACE::Queue& active_queue) ///< [in] The queue to be associated with this allocation
    {
      // CMS: use RAII instead of (un)locking explicitly
      std::unique_lock<std::mutex> mutex_locker(mutex, std::defer_lock);
      size_t bytes = sizeof(TData) * extent;
      
      // Create a block descriptor for the requested allocation
      bool found = false;
      BlockDescriptor search_key(device, active_queue);
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
        mutex_locker.lock();

        if (search_key.bin < min_bin) {
          // Bin is less than minimum bin: round up
          search_key.bin = min_bin;
          search_key.bytes = min_bin_bytes;
        }

        // Iterate through the range of cached blocks on the same device in the same bin
        auto block_itr = cached_blocks.lower_bound(search_key);
        while ((block_itr != cached_blocks.end()) && (block_itr->device == device) &&
               (block_itr->bin == search_key.bin)) {
          // To prevent races with reusing blocks returned by the host but still
          // in use by the device, only consider cached blocks that are
          // either (from the active queue) or (from an idle queue)
          if ((active_queue == block_itr->associated_queue) ||
              (::alpaka::isComplete(block_itr->ready_event))) {
            // Reuse existing cache block.  Insert into live blocks.
            found = true;
            search_key = *block_itr;
            search_key.associated_queue = active_queue;
            live_blocks.insert(search_key);

            // Remove from free blocks
            cached_bytes[device].free -= search_key.bytes;
            cached_bytes[device].live += search_key.bytes;
            cached_bytes[device].liveRequested += search_key.bytesRequested;  // CMS

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

            break;
          }
          block_itr++;
        }

        // Done searching: unlock
        mutex_locker.unlock();
      }

      // Allocate the block if necessary
      if (!found) {
        search_key.buf = cms::alpakatools::allocDeviceBuf<TData>(extent);

        // Create ready event
        search_key.ready_event = ::alpaka::Event<ALPAKA_ACCELERATOR_NAMESPACE::Queue> {device};

        // Insert into live blocks
        mutex_locker.lock();
        live_blocks.insert(search_key);
        cached_bytes[device].live += search_key.bytes;
        cached_bytes[device].liveRequested += search_key.bytesRequested;  // CMS
        mutex_locker.unlock();

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

      return search_key.buf;  
    }

    /**
     * \brief Provides a suitable allocation of device memory for the given size on the current device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_queue
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * queues when all prior work submitted to \p active_queue has completed.
     */
    auto DeviceAllocate(
        const alpaka_common::Extent& extent,                     ///< [in] Extent of the allocation
        const ALPAKA_ACCELERATOR_NAMESPACE::Queue& active_queue) ///< [in] The queue to be associated with this allocation
    {
      return DeviceAllocate(ALPAKA_ACCELERATOR_NAMESPACE::device, extent, active_queue);
    }

    /**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_queue
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * queues when all prior work submitted to \p active_queue has completed.
     */
    void DeviceFree(
      const ALPAKA_ACCELERATOR_NAMESPACE::DevAcc1& device, 
      ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> &buf) 
    {
      // CMS: use RAII instead of (un)locking explicitly
      std::unique_lock<std::mutex> mutex_locker(mutex, std::defer_lock);

      // Lock
      mutex_locker.lock();

      // Find corresponding block descriptor
      bool recached = false;
      BlockDescriptor search_key(std::move(buf), device);
      auto block_itr = live_blocks.find(search_key);
      if (block_itr != live_blocks.end()) {
        // Remove from live blocks
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_bytes[device].live -= search_key.bytes;
        cached_bytes[device].liveRequested -= search_key.bytesRequested;  // CMS

        // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
        if ((search_key.bin != INVALID_BIN) && (cached_bytes[device].free + search_key.bytes <= max_cached_bytes)) {
          // Insert returned allocation into free blocks
          recached = true;
          cached_blocks.insert(search_key);
          cached_bytes[device].free += search_key.bytes;

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

      if (recached) {
        // Insert the ready event in the associated queue
        ::alpaka::enqueue(search_key.associated_queue, search_key.ready_event);
      }

      // Unlock
      mutex_locker.unlock();

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

    /**
     * \brief Frees a live allocation of device memory on the current device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_queue
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * queues when all prior work submitted to \p active_queue has completed.
     */
    void DeviceFree(ALPAKA_ACCELERATOR_NAMESPACE::AlpakaDeviceBuf<TData> &buf) { 
      return DeviceFree(ALPAKA_ACCELERATOR_NAMESPACE::device, buf); 
    }

    /**
     * \brief Frees all cached device allocations on all devices
     */
    void FreeAllCached() {
      // CMS: use RAII instead of (un)locking explicitly
      std::unique_lock<std::mutex> mutex_locker(mutex);

      while (!cached_blocks.empty()) {
        // Get first block
        auto begin = cached_blocks.begin();

        // Reduce balance and erase entry
        cached_bytes[begin->device].free -= begin->bytes;

        /*
        if (debug)
          printf(
              "\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld "
              "bytes) outstanding.\n",
              current_device,
              (long long)begin->bytes,
              (long long)cached_blocks.size(),
              (long long)cached_bytes[current_device].free,
              (long long)live_blocks.size(),
              (long long)cached_bytes[current_device].live);
        */

        cached_blocks.erase(begin);
      }

      mutex_locker.unlock();
    }

    // CMS: give access to cache allocation status
    DeviceCachedBytes CacheStatus() const {
      std::unique_lock mutex_locker(mutex);
      return cached_bytes;
    }

    /**
     * \brief Destructor
     */
    // CMS: make the destructor not virtual
    ~CachingDeviceAllocator() {
      if (!skip_cleanup)
        FreeAllCached();
    }
  };

  /** @} */  // end group UtilMgmt

}  // namespace cms::alpaka::allocator

#endif
