#ifndef VOXBLOX_OBJECT_POOL_H_
#define VOXBLOX_OBJECT_POOL_H_

#include <array>
#include <deque>
#include <cmath>
#include <unordered_set>

#include <glog/logging.h>

namespace voxblox_fast {
template<typename ObjectType, size_t kNumAlignedBytes = 8,
    size_t kMemoryChunkSizeBytes = 1024u * 1024u>
class ObjectPool {
 public:
  static_assert(sizeof(ObjectType) < kMemoryChunkSizeBytes,
                "Object must be smaller than the memory chunks.");

  template<class... Args>
      ObjectType* AllocateObject(Args&&... constructor_args) {
    ObjectType* object = nullptr;
    // Find a memory chunk that has space; otherwise allocate a new chunk.
    for (MemoryChunk& memory_chunk : memory_chunks_) {
      if ((object = memory_chunk.AllocateObject(constructor_args...))) {
        return object;
      }
    }

    // All chunks are full, let's allocate a new one.
    MemoryChunk* new_chunk = AllocateNewMemoryChunk();
    object = new_chunk->AllocateObject(constructor_args...);
    CHECK(object != nullptr);
    return object;
  }

  void DeallocateObject(const ObjectType* element) {
    const size_t chunk_index = GetIndexOfStoringMemoryChunk(element);
    memory_chunks_[chunk_index].DeallocateObject(element);
    if (memory_chunks_[chunk_index].IsEmpty()) {
      DeallocateMemoryChunk(chunk_index);
    }
  }

  size_t GetNumMemoryChunks() const {
    return memory_chunks_.size();
  }

 private:
  struct MemoryChunk {
   public:
    static constexpr size_t GetSlotSize()  {
      return std::ceil(
          static_cast<double>(sizeof(ObjectType)) / kNumAlignedBytes) *
            kNumAlignedBytes;
    }
    static constexpr size_t kSlotSizeBytes = GetSlotSize();
    static constexpr size_t GetNumSlots() {
        return kMemoryChunkSizeBytes / GetSlotSize();
    }
    static constexpr size_t kNumSlots = GetNumSlots();
    static_assert(kNumSlots > 0, "Data does not fit a memory chunk.");

    ~MemoryChunk() {
      // Call the dtor of all objects.
      for (int slot_index = 0u; slot_index < kNumSlots; ++slot_index) {
        const bool slot_is_set = slot_index < next_free_slot_index &&
            freed_slot_indices.count(slot_index) == 0;
        if (slot_is_set) {
          DeallocateObject(slot_index);
        }
      }
      CHECK(IsEmpty());
    }

    size_t NumUsedSlots() const {
      return next_free_slot_index - freed_slot_indices.size();
    }

    bool IsEmpty() const {
      return NumUsedSlots() == 0;
    }

    bool IsFull() const {
      return NumUsedSlots() >= kNumSlots;
    }

    // Returns nullptr if chunk is full.
    template<class... Args>
    ObjectType* AllocateObject(Args&&... constructor_args) {
      // Try to get a slot by linearly filling the memory.
      if (next_free_slot_index < kNumSlots) {
        const size_t slot_idx = next_free_slot_index;
        ++next_free_slot_index;
        return new (&memory[slot_idx * kSlotSizeBytes]) ObjectType(
            constructor_args...);
      }

      // Next try to use a slot that was released.
      if (!freed_slot_indices.empty()) {
        const size_t slot_idx = *freed_slot_indices.begin();
        freed_slot_indices.erase(freed_slot_indices.begin());
        return new (&memory[slot_idx * kSlotSizeBytes]) ObjectType(
            constructor_args...);
      }

      // Memory is full.
      return nullptr;
    }

    void DeallocateObject(size_t slot_idx) {
      GetObjectAddressFromSlotIndex(slot_idx)->~ObjectType();
      CHECK(freed_slot_indices.emplace(slot_idx).second);
    }

    void DeallocateObject(const ObjectType* object) {
      object->~ObjectType();
      CHECK(freed_slot_indices.emplace(
          GetSlotIndexOfObjectAddress(object)).second);
    }

    size_t GetSlotIndexOfObjectAddress(const ObjectType* const object) const {
      const char* const object_char = reinterpret_cast<const char*>(object);
      const char* const chunk_start = &memory[0];
      CHECK_GE(object_char, chunk_start);
      const size_t slot_index = (object_char - chunk_start) / kSlotSizeBytes;
      CHECK_LT(slot_index, kNumSlots);
      return slot_index;
    }

    ObjectType* GetObjectAddressFromSlotIndex(size_t slot_index) {
      CHECK_LT(slot_index, kNumSlots);
      return reinterpret_cast<ObjectType*>(&memory[0] +
          slot_index * kSlotSizeBytes);
    }

    alignas(kNumAlignedBytes) std::array<char, kMemoryChunkSizeBytes> memory;
    std::unordered_set<size_t> freed_slot_indices;
    size_t next_free_slot_index;
  };

  MemoryChunk* AllocateNewMemoryChunk() {
    memory_chunks_.emplace_back();
    return &memory_chunks_.back();
  }

  void DeallocateMemoryChunk(size_t chunk_index) {
    CHECK_LT(chunk_index, memory_chunks_.size());
    memory_chunks_.erase(memory_chunks_.begin() + chunk_index);
  }

  size_t GetIndexOfStoringMemoryChunk(const ObjectType* object) const {
    for (size_t chunk_idx = 0u; chunk_idx < memory_chunks_.size();
        ++chunk_idx) {
      const char* const address_first_element =
          &memory_chunks_[chunk_idx].memory.front();
      if (object < address_first_element) {
        // Object lies below this chunks address range.
        continue;
      }

      const char* const address_last_element =
          address_first_element + MemoryChunk::kNumSlots;
      if (object < address_last_element) {
        // Object lies within this chunks address range.
        return chunk_idx;
      };
    }
    LOG(FATAL) << "We should never reach here.";
  }

  std::deque<MemoryChunk> memory_chunks_;
};

}  // namespace voxblox

#endif  // VOXBLOX_OBJECT_POOL_H_
