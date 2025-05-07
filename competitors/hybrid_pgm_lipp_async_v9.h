#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include "pgm_index.h"

#include "../searches/linear_search.h"
#include "../searches/branching_binary_search.h"
#include "../searches/interpolation_search.h"
#include "../searches/exponential_search.h"

/**
 * Single‑thread benchmark version with background flusher.
 * Fixes:
 *  - No destructor override (base has no virtual dtor).
 *  - Removes call to non‑existent LIPP::update(); duplicates are skipped.
 *  - Prevents duplicate inserts that triggered bitmap assertion.
 */

template <class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPPAsync : public Competitor<KeyType, SearchClass> {
public:
  explicit HybridPGMLIPPAsync(const std::vector<int>& params) {
    flush_threshold_ = params.empty() ? 10'000 : static_cast<size_t>(params[0]);
    const size_t ks = sizeof(KeyType);
    if (ks <= 4)  flush_threshold_ = std::max(flush_threshold_, 20'000UL);
    if (ks >= 16) flush_threshold_ = std::min(flush_threshold_,  5'000UL);
    start_flusher_thread();
  }

  ~HybridPGMLIPPAsync() {
    stop_flag_.store(true, std::memory_order_release);
    cv_.notify_one();
    if (flusher_thread_.joinable()) flusher_thread_.join();
    drain_buffer_to_lipp();
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t /*num_threads*/) {
    std::vector<std::pair<KeyType, uint64_t>> base;
    base.reserve(data.size());
    for (const auto& kv : data) base.emplace_back(kv.key, kv.value);

    if (data.size() > 1'000'000)
      flush_threshold_ = std::max(flush_threshold_, data.size() / 100);

    return util::timing([&] {
      pgm_ = decltype(pgm_)(base.begin(), base.end());
      if (base.size() >= MIN_BULK_SIZE) {
        std::sort(base.begin(), base.end(), [](auto& a, auto& b) { return a.first < b.first; });
        lipp_.bulk_load(base.data(), base.size());
        lipp_initialized_.store(true, std::memory_order_release);
      }
    });
  }

  void Insert(const KeyValue<KeyType>& kv, uint32_t /*thread_id*/) {
    pgm_.insert(kv.key, kv.value);
    {
      std::lock_guard<std::mutex> lg(buffer_mtx_);
      buffer_[kv.key] = kv.value;
      if (++insert_count_ >= flush_threshold_) cv_.notify_one();
    }
  }

  size_t EqualityLookup(const KeyType& key, uint32_t /*thread_id*/) const {
    {
      std::lock_guard<std::mutex> lg(buffer_mtx_);
      auto it = buffer_.find(key);
      if (it != buffer_.end()) return it->second;
    }
    {
      std::shared_lock<std::shared_mutex> sl(lipp_mtx_);
      if (lipp_initialized_) {
        uint64_t val;
        if (lipp_.find(key, val)) return val;
      }
    }
    auto it = pgm_.find(key);
    return it != pgm_.end() ? it->value() : util::NOT_FOUND;
  }

  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t /*thread_id*/) const {
    uint64_t sum = 0;
    std::unordered_map<KeyType, bool> seen;

    {
      std::lock_guard<std::mutex> lg(buffer_mtx_);
      for (auto& [k, v] : buffer_)
        if (k >= lo && k <= hi) {
          sum += v;
          seen.emplace(k, true);
        }
    }
    {
      std::shared_lock<std::shared_mutex> sl(lipp_mtx_);
      if (lipp_initialized_) {
        auto it = lipp_.lower_bound(lo);
        while (it != lipp_.end() && it->comp.data.key <= hi) {
          if (!seen.count(it->comp.data.key)) sum += it->comp.data.value;
          ++it;
        }
      }
    }
    auto pit = pgm_.lower_bound(lo);
    while (pit != pgm_.end() && pit->key() <= hi) {
      if (!seen.count(pit->key())) sum += pit->value();
      ++pit;
    }
    return sum;
  }

  bool applicable(bool /*unique*/, bool /*range*/, bool /*insert*/, bool multithread,
                  const std::string& /*ops*/) const {
    return SearchClass::name() != "LinearAVX" && !multithread;
  }

  std::string name() const { return "HybridPGMLIPPASYNC"; }

  std::vector<std::string> variants() const {
    return {SearchClass::name(), std::to_string(pgm_error),
            "FlushThreshold=" + std::to_string(flush_threshold_)};
  }

  std::size_t size() const {
    size_t sz = pgm_.size_in_bytes();
    if (lipp_initialized_) sz += lipp_.index_size();
    sz += buffer_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    return sz;
  }

private:
  /* ---- flushing helpers ---- */
  void start_flusher_thread() {
    flusher_thread_ = std::thread([this] {
      std::unique_lock<std::mutex> ul(buffer_mtx_);
      while (!stop_flag_.load(std::memory_order_acquire)) {
        cv_.wait(ul, [this] {
          return stop_flag_.load(std::memory_order_relaxed) || buffer_.size() >= flush_threshold_;
        });
        if (stop_flag_) break;
        ul.unlock();
        drain_buffer_to_lipp();
        ul.lock();
      }
    });
  }

  void drain_buffer_to_lipp() {
    std::unordered_map<KeyType, uint64_t> snap;
    {
      std::lock_guard<std::mutex> lg(buffer_mtx_);
      if (buffer_.empty()) return;
      snap.swap(buffer_);
      insert_count_ = 0;
    }

    std::vector<std::pair<KeyType, uint64_t>> vec;
    vec.reserve(snap.size());
    for (auto& kv : snap) vec.emplace_back(kv);

    std::unique_lock<std::shared_mutex> ul(lipp_mtx_);
    if (!lipp_initialized_ && vec.size() >= MIN_BULK_SIZE) {
      std::sort(vec.begin(), vec.end(), [](auto& a, auto& b) { return a.first < b.first; });
      lipp_.bulk_load(vec.data(), vec.size());
      lipp_initialized_.store(true, std::memory_order_release);
      return;
    }
    if (!lipp_initialized_) return; // too small for first bulk load

    for (auto& kv : vec) {
      uint64_t dummy;
      if (lipp_.find(kv.first, dummy)) continue; // duplicate, skip
      lipp_.insert(kv.first, kv.second);
    }
  }

  /* ---- members ---- */
  static constexpr size_t MIN_BULK_SIZE = 1'000;

  DynamicPGMIndex<KeyType, uint64_t, SearchClass,
                  PGMIndex<KeyType, SearchClass, pgm_error, 16>> pgm_;
  LIPP<KeyType, uint64_t> lipp_;

  mutable std::mutex buffer_mtx_;
  std::unordered_map<KeyType, uint64_t> buffer_;
  std::atomic<size_t> insert_count_{0};

  size_t                       flush_threshold_;
  std::condition_variable      cv_;
  std::thread                  flusher_thread_;
  std::atomic<bool>            stop_flag_{false};

  mutable std::shared_mutex    lipp_mtx_;
  std::atomic<bool>            lipp_initialized_{false};
};

#endif /* TLI_HYBRID_PGM_LIPP_ASYNC_H */
