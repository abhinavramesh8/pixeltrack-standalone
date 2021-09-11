#ifndef HeterogenousCore_AlpakaUtilities_src_CachingHostAllocator_h
#define HeterogenousCore_AlpakaUtilities_src_CachingHostAllocator_h

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
 * Modified to cache pinned host allocations by Matti Kortelainen
 */

/******************************************************************************
 * Simple caching allocator for pinned host memory allocations. The allocator is
 * thread-safe.
 ******************************************************************************/

#include <cmath>
#include <memory>
#include <set>
#include <mutex>

#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaCore/deviceAllocatorStatus.h"

/// cms::alpaka::allocator namespace
namespace cms::alpakatools::allocator {

  /**
 * \addtogroup UtilMgmt
 * @{
 */

  /******************************************************************************
 * CachingHostAllocator (host use)
 ******************************************************************************/

  /**
 * \brief A simple caching allocator pinned host memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe.  It behaves as follows:
 *
 * To read/write from/to the pinned host memory, one needs to synchronize 
 * anyway. The difference wrt. device memory is that in the CPU all 
 * operations to the device memory are scheduled via the queue, while 
 * for the host memory one can perform operations directly.
 *
 * \par
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused host allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations  will exceed
 *   \p max_cached_bytes, allocations are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingHostAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes
 *
 */
  // template <typename TData>
  struct CachingHostAllocator {
    //---------------------------------------------------------------------
    // Constants
    //---------------------------------------------------------------------

    /// Out-of-bounds bin
    static const unsigned int INVALID_BIN = (unsigned int)-1;

    /// Invalid size
    static const size_t INVALID_SIZE = (size_t)-1;

#ifndef DOXYGEN_SHOULD_SKIP_THIS  // Do not document

    /// Invalid device ordinal
    // static const int INVALID_DEVICE_ORDINAL = -1;

    //---------------------------------------------------------------------
    // Type definitions and helper types
    //---------------------------------------------------------------------

    /**
     * Descriptor for pinned host memory allocations
     */
    struct BlockDescriptor {
      void* d_ptr; // Native host pointer
      std::shared_ptr<alpaka_common::AlpakaHostBuf<std::byte>> buf_ptr; // Host buffer
      size_t bytes; // Size of allocation in bytes
      unsigned int bin; // Bin enumeration
      // int device_idx; // Device
      // ::shared_ptr<ALPAKA_ACCELERATOR_NAMESPACE::Queue> associated_queue_ptr; // Associated associated_queue
      // std::shared_ptr<alpaka::Event<ALPAKA_ACCELERATOR_NAMESPACE::Queue>> ready_event_ptr; // Signal when associated queue has run to the point at which this block was freed

      // Constructor (suitable for searching maps for a specific block, given its native host pointer)
      BlockDescriptor(void* ptr)
          : d_ptr(ptr),
            buf_ptr(nullptr),
            bytes(0),
            bin(INVALID_BIN)/*,
            device_idx(INVALID_DEVICE_ORDINAL),
            associated_queue_ptr(nullptr),
            ready_event_ptr(nullptr)*/ {}

      // Constructor (suitable for searching maps for a range of suitable blocks)
      /*BlockDescriptor(int dev_idx)
          : d_ptr(nullptr),
            buf_ptr(nullptr),
            bytes(0),
            bin(INVALID_BIN),
            device_idx(dev_idx),
            associated_queue_ptr(nullptr),
            ready_event_ptr(nullptr) {}*/

      // Comparison functor for comparing host pointers
      static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b) {
        return (a.d_ptr < b.d_ptr); 
      }

      // Comparison functor for comparing allocation sizes
      static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b) { return (a.bytes < b.bytes); }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

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

    std::mutex mutex;  /// Mutex for thread-safety

    unsigned int bin_growth;  /// Geometric growth factor for bin-sizes
    unsigned int min_bin;     /// Minimum bin enumeration
    unsigned int max_bin;     /// Maximum bin enumeration

    size_t min_bin_bytes;     /// Minimum bin size
    size_t max_bin_bytes;     /// Maximum bin size
    size_t max_cached_bytes;  /// Maximum aggregate cached bytes

    bool debug;        /// Whether or not to print (de)allocation events to stdout

    TotalBytes cached_bytes;     /// Aggregate cached bytes
    CachedBlocks cached_blocks;  /// Set of cached pinned host allocations available for reuse
    BusyBlocks live_blocks;      /// Set of live pinned host allocations currently in use

#endif  // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingHostAllocator(
        unsigned int bin_growth,                 ///< Geometric growth factor for bin-sizes
        unsigned int min_bin = 1,                ///< Minimum bin (default is bin_growth ^ 1)
        unsigned int max_bin = INVALID_BIN,      ///< Maximum bin (default is no max bin)
        size_t max_cached_bytes = INVALID_SIZE,  ///< Maximum aggregate cached bytes (default is no limit)
        bool debug = false)  ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
        : bin_growth(bin_growth),
          min_bin(min_bin),
          max_bin(max_bin),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes(max_cached_bytes),
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
     * sets a maximum of 6,291,455 cached bytes
     */
    CachingHostAllocator(/*bool skip_cleanup = false, */bool debug = false)
        : bin_growth(8),
          min_bin(3),
          max_bin(7),
          min_bin_bytes(IntPow(bin_growth, min_bin)),
          max_bin_bytes(IntPow(bin_growth, max_bin)),
          max_cached_bytes((max_bin_bytes * 3) - 1),
          debug(debug),
          cached_blocks(BlockDescriptor::SizeCompare),
          live_blocks(BlockDescriptor::PtrCompare) {}

    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache
     *
     * Changing the ceiling of cached bytes does not cause any allocations (in-use or
     * cached-in-reserve) to be freed.  See \p FreeAllCached().
     */
    void SetMaxCachedBytes(size_t max_cached_bytes) {
      // Lock
      // std::unique_lock mutex_locker(mutex);
      mutex.lock();

      if (debug)
        printf("Changing max_cached_bytes (%lld -> %lld)\n",
               (long long)this->max_cached_bytes,
               (long long)max_cached_bytes);

      this->max_cached_bytes = max_cached_bytes;

      // Unlock (redundant, kept for style uniformity)
      mutex.unlock();
    }

    /**
     * \brief Provides a suitable allocation of pinned host memory for the given size.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    template <typename TData>
    auto HostAllocate(
        const alpaka_common::Extent& extent,                     ///< [in] Extent of the allocation
        const ALPAKA_ACCELERATOR_NAMESPACE::Queue& active_queue) ///< [in] The queue to be associated with this allocation
    {
      // std::unique_lock<std::mutex> mutex_locker(mutex, std::defer_lock);
      // auto device = alpaka::getDev(active_queue);
      // auto device_idx = getIdxOfDev(device);
      size_t bytes = cms::alpakatools::nbytesFromExtent<TData>(extent);

      // Create a block descriptor for the requested allocation
      bool found = false;
      BlockDescriptor search_key(nullptr/*device_idx*/);
      // auto active_queue_ptr = std::make_shared<ALPAKA_ACCELERATOR_NAMESPACE::Queue>(active_queue);
      // auto ready_event_ptr = std::make_shared<alpaka::Event<ALPAKA_ACCELERATOR_NAMESPACE::Queue>>(device);
      // search_key.associated_queue_ptr = active_queue_ptr;
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

        // Iterate through the range of cached blocks in the same bin
        auto block_itr = cached_blocks.find(search_key);
        if (block_itr != cached_blocks.end()) {
          // To prevent races with reusing blocks returned by the host but still
          // in use for transfers, only consider cached blocks that are from an idle queue
          // if ((block_itr->ready_event_ptr == nullptr) || alpaka::isComplete(*(block_itr->ready_event_ptr))) {
            // Reuse existing cache block.  Insert into live blocks.
          found = true;
          search_key = *block_itr;
          // search_key.associated_queue_ptr = active_queue_ptr;
          /*if (search_key.device_idx != device_idx) {
            // If "associated" device changes, need to re-create the event on the right device
            search_key.ready_event_ptr = ready_event_ptr;
            search_key.device_idx = device_idx;
          }*/

          live_blocks.insert(search_key);

          // Remove from free blocks
          cached_bytes.free -= search_key.bytes;
          cached_bytes.live += search_key.bytes;

          /*if (debug)
            printf(
                "\tHost reused cached block at %p (%lld bytes) for queue %lld, event %lld on device %lld "
                "(previously associated with queue %lld, event %lld).\n",
                search_key.d_ptr,
                (long long)search_key.bytes,
                search_key.associated_queue,
                search_key.ready_event,
                search_key.device,
                (long long)block_itr->associated_queue,
                (long long)block_itr->ready_event);*/

          cached_blocks.erase(block_itr);

          //   break;
          // }
          // block_itr++;
        }

        // Done searching: unlock
        mutex.unlock();
      }

      // Allocate the block if necessary
      if (!found) {
        // Attempt to allocate
        // TODO: eventually support allocation flags
        auto buf {cms::alpakatools::allocHostBuf<std::byte>(
          static_cast<alpaka_common::Extent>(search_key.bytes))};
        alpaka::prepareForAsyncCopy(buf);
        search_key.d_ptr = alpaka::getPtrNative(buf);
        search_key.buf_ptr = std::make_shared<alpaka_common::AlpakaHostBuf<std::byte>>(
          std::move(buf)
        );
        // search_key.ready_event_ptr = ready_event_ptr;
        
        // Insert into live blocks
        mutex.lock();
        live_blocks.insert(search_key);
        cached_bytes.live += search_key.bytes;
        mutex.unlock();

        /*
        if (debug)
          printf(
              "\tHost allocated new host block at %p (%lld bytes associated with queue %lld, event %lld on device "
              "%lld).\n",
              search_key.d_ptr,
              (long long)search_key.bytes,
              search_key.associated_queue,
              search_key.ready_event,
              search_key.device);
        */
      }

      /*
      if (debug)
        printf("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
               (long long)cached_blocks.size(),
               (long long)cached_bytes.free,
               (long long)live_blocks.size(),
               (long long)cached_bytes.live);
      */

      return search_key.buf_ptr.get();
    }

    /**
     * \brief Frees a live allocation of pinned host memory, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    void HostFree(void* d_ptr) {
      // Lock
      // std::unique_lock<std::mutex> mutex_locker(mutex);
      mutex.lock();

      // Find corresponding block descriptor
      BlockDescriptor search_key(d_ptr);
      auto block_itr = live_blocks.find(search_key);
      if (block_itr != live_blocks.end()) {
        // Remove from live blocks
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_bytes.live -= search_key.bytes;

        // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
        if ((search_key.bin != INVALID_BIN) && (cached_bytes.free + search_key.bytes <= max_cached_bytes)) {
          // Insert returned allocation into free blocks
          cached_blocks.insert(search_key);
          cached_bytes.free += search_key.bytes;
          /*if (search_key.associated_queue_ptr && search_key.ready_event_ptr) {
            alpaka::enqueue(*(search_key.associated_queue_ptr), *(search_key.ready_event_ptr));
          }*/
          /*
          if (debug)
            printf(
                "\tHost returned %lld bytes from associated queue %lld, event %lld on device %lld.\n\t\t %lld "
                "available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                (long long)search_key.bytes,
                search_key.associated_queue,
                search_key.ready_event,
                search_key.device,
                (long long)cached_blocks.size(),
                (long long)cached_bytes.free,
                (long long)live_blocks.size(),
                (long long)cached_bytes.live);
          */
        }
      }

      // Unlock
      mutex.unlock();

      /*
      if (!recached and debug)
        printf(
            "\tHost freed %lld bytes from associated queue %lld, event %lld on device %lld.\n\t\t  %lld available "
            "blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
            (long long)search_key.bytes,
            search_key.associated_queue,
            search_key.ready_event,
            search_key.device,
            (long long)cached_blocks.size(),
            (long long)cached_bytes.free,
            (long long)live_blocks.size(),
            (long long)cached_bytes.live);
      */
    }

  };

  /** @} */  // end group UtilMgmt

} // namespace cms::alpakatools::allocator

#endif
