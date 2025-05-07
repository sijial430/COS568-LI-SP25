#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

/*
 * HybridPGMLIPPAsync – multi‑shard (lock‑free, single thread).
 * ===========================================================
 * Implementation core lives in template **HybridPGMLIPPAsyncImpl**.
 * For compatibility with the benchmark harness that expects the
 * three‑parameter template <KeyType, SearchClass, pgm_error>, we
 * provide an alias **HybridPGMLIPPAsync** at the end of the file.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include <string>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include "pgm_index.h"
#include "../searches/branching_binary_search.h"

/* ------------------------------------------------------------------ */

template <class  KeyType,
          size_t NumShards    = 8,          /* power‑of‑two, >=1 */
          size_t pgm_error    = 32,         /* PGM segment error */
          size_t kBufCap      = 1 << 10,    /* 1 024 staged keys */
          size_t kFlushChunk  = 128,        /* migrate per insert*/
          class  SearchClass  = BranchingBinarySearch<0>>
class HybridPGMLIPPAsyncImpl : public Competitor<KeyType, SearchClass> {
  static_assert(std::is_integral_v<KeyType>, "KeyType must be integral");
  static_assert((NumShards & (NumShards - 1)) == 0, "NumShards must be power‑of‑two");
public:
  explicit HybridPGMLIPPAsyncImpl(const std::vector<int>&) {}

  /* ---------- build -------------------------------------------- */
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t) {
    std::array<std::vector<std::pair<KeyType,uint64_t>>, NumShards> bucket;
    for (auto& vec : bucket) vec.clear();
    for (const auto& kv : data)
      bucket[shard_of(kv.key)].push_back({kv.key, kv.value});

    return util::timing([&] {
      for (size_t s = 0; s < NumShards; ++s) {
        auto& vec = bucket[s];
        std::sort(vec.begin(), vec.end(), [](auto&a,auto&b){return a.first<b.first;});
        pgm_[s] = Pgm(vec.begin(), vec.end());
        if (vec.size() >= MIN_BULK_SIZE) {
          lipp_[s].bulk_load(vec.data(), vec.size());
          lipp_init_[s] = true;
        }
      }
    });
  }

  /* ---------- insert ------------------------------------------- */
  void Insert(const KeyValue<KeyType>& kv, uint32_t) {
    const size_t s = shard_of(kv.key);
    pgm_[s].insert(kv.key, kv.value);

    const size_t idx = tail_[s] & (kBufCap - 1);
    buf_key_[s][idx] = kv.key;
    buf_val_[s][idx] = kv.value;
    ++tail_[s]; ++staged_[s];

    if (staged_[s] >= kFlushChunk) flush_chunk(s);
  }

  /* ---------- equality lookup ---------------------------------- */
  size_t EqualityLookup(const KeyType& key, uint32_t) const {
    const size_t s = shard_of(key);

    for (size_t i = 0; i < staged_[s]; ++i) {
      const size_t idx = (head_[s] + i) & (kBufCap - 1);
      if (buf_key_[s][idx] == key) return buf_val_[s][idx];
    }
    if (lipp_init_[s]) {
      uint64_t v; if (lipp_[s].find(key, v)) return v;
    }
    auto it = pgm_[s].find(key);
    return it != pgm_[s].end() ? it->value() : util::NOT_FOUND;
  }

  /* ---------- range query -------------------------------------- */
  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t) const {
    uint64_t sum = 0;
    for (size_t s = 0; s < NumShards; ++s) {
      for (size_t i = 0; i < staged_[s]; ++i) {
        const size_t idx = (head_[s] + i) & (kBufCap - 1);
        const KeyType k = buf_key_[s][idx];
        if (k >= lo && k <= hi) sum += buf_val_[s][idx];
      }
      if (lipp_init_[s]) {
        auto it = lipp_[s].lower_bound(lo);
        while (it != lipp_[s].end() && it->comp.data.key <= hi) {
          sum += it->comp.data.value; ++it;
        }
      }
      auto pit = pgm_[s].lower_bound(lo);
      while (pit != pgm_[s].end() && pit->key() <= hi) { sum += pit->value(); ++pit; }
    }
    return sum;
  }

  /* ---------- benchmark meta ----------------------------------- */
  bool applicable(bool /*unique*/, bool /*range_query*/, bool /*insert*/,
                  bool /*multithread*/, const std::string& /*ops_filename*/) const {
    return true; /* allow multi-thread */
  }
  std::string name() const { return "HybridPGMLIPPASYNC"; }
  std::vector<std::string> variants() const {
    return {SearchClass::name(), std::to_string(pgm_error), "Shards="+std::to_string(NumShards), "kFlushChunk="+std::to_string(kFlushChunk), "kBufCap="+std::to_string(kBufCap)};
  }
  std::size_t size() const {
    size_t sz = 0;
    for (size_t s=0;s<NumShards;++s) {
      sz += pgm_[s].size_in_bytes();
      if (lipp_init_[s]) sz += lipp_[s].index_size();
    }
    sz += NumShards * kBufCap * (sizeof(KeyType)+sizeof(uint64_t));
    return sz;
  }

private:
  /* ---------- helper routines ---------------------------------- */
  static constexpr size_t MIN_BULK_SIZE = 16'384;
  static inline size_t shard_of(KeyType k) noexcept { return static_cast<size_t>(k) & (NumShards - 1); }

  void flush_chunk(size_t s) {
    uint64_t tmp;
    const size_t n = std::min(staged_[s], kFlushChunk);
    for (size_t i = 0; i < n; ++i) {
      const size_t idx = head_[s] & (kBufCap - 1);
      const KeyType  k = buf_key_[s][idx];
      const uint64_t v = buf_val_[s][idx];
      ++head_[s]; --staged_[s];

      if (lipp_init_[s]) {
        if (lipp_[s].find(k, tmp)) continue;           /* duplicate */
        lipp_[s].insert(k, v);
        pgm_[s].erase(k);
      } else {
        scratch_[s].push_back({k, v});
        if (scratch_[s].size() >= MIN_BULK_SIZE) {
          std::sort(scratch_[s].begin(), scratch_[s].end(), [](auto&a,auto&b){return a.first<b.first;});
          lipp_[s].bulk_load(scratch_[s].data(), scratch_[s].size());
          scratch_[s].clear(); lipp_init_[s] = true;
        }
      }
    }
  }

  /* ---------- type aliases ------------------------------------- */
  using Pgm = DynamicPGMIndex<KeyType, uint64_t, SearchClass,
                              PGMIndex<KeyType, SearchClass, pgm_error, 16>>;
  using Lipp = LIPP<KeyType, uint64_t>;

  /* ---------- per‑shard data ----------------------------------- */
  std::array<Pgm,  NumShards> pgm_{};
  std::array<Lipp, NumShards> lipp_{};
  std::array<bool, NumShards> lipp_init_{};

  std::array<std::array<KeyType , kBufCap>,  NumShards> buf_key_{};
  std::array<std::array<uint64_t, kBufCap>, NumShards> buf_val_{};
  std::array<size_t, NumShards> head_{};
  std::array<size_t, NumShards> tail_{};
  std::array<size_t, NumShards> staged_{};

  std::array<std::vector<std::pair<KeyType,uint64_t>>, NumShards> scratch_{};
};

/* ---------- compatibility alias for benchmark harness ---------- */
template <class KeyType, class SearchClass, size_t pgm_error>
using HybridPGMLIPPAsync = HybridPGMLIPPAsyncImpl<KeyType, 4, pgm_error, 1<<10, 128, SearchClass>;

#endif /* TLI_HYBRID_PGM_LIPP_ASYNC_H */