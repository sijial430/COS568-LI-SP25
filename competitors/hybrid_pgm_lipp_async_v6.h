#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index_async.h"
#include "lipp_async.h"
#include "pgm_index.h"
#include "../searches/branching_binary_search.h"
#include "../searches/exponential_search.h"
#include "../searches/interpolation_search.h"
#include "../searches/linear_search.h"

template <class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPPAsync : public Competitor<KeyType, SearchClass> {
 public:
  explicit HybridPGMLIPPAsync(const std::vector<int>& params) {
    // Initial buffer‐capacity hint
    size_t init_cap = params.empty() ? 200'000 : params[0];
    buffer_cap_.store(init_cap);

    stop_flag_.store(false);
    insert_ops_.store(0);
    lookup_ops_.store(0);

    const size_t hw = std::thread::hardware_concurrency();
    num_flush_threads_ = std::max<size_t>(1, hw / 2);
    for (size_t i = 0; i < num_flush_threads_; ++i)
      flush_threads_.emplace_back([this] { FlushThreadFunc(); });
  }

  ~HybridPGMLIPPAsync() {
    /* 1. stop workers */
    stop_flag_.store(true);
    flush_cv_.notify_all();
    for (auto& t : flush_threads_)
      if (t.joinable()) t.join();

    /* 2. drain pending queue */
    while (true) {
      std::vector<std::pair<KeyType, uint64_t>> batch;
      {
        std::lock_guard<std::mutex> ql(flush_queue_mutex_);
        if (flush_queue_.empty()) break;
        batch = std::move(flush_queue_.front());
        flush_queue_.pop();
      }
      process_batch_into_lipp(batch);
    }

    /* 3. final flush of D-PGM */
    std::vector<std::pair<KeyType, uint64_t>> tail;
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      tail.reserve(dpgm_.size());
      for (auto it = dpgm_.begin(); it != dpgm_.end(); ++it)
        tail.emplace_back(it->key(), it->value());
      for (auto& kv : tail) dpgm_.erase(kv.first);
    }
    if (!tail.empty()) process_batch_into_lipp(tail);
  }

  /* ---------- offline build ---------- */
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t) {
    std::vector<std::pair<KeyType, uint64_t>> base;
    base.reserve(data.size());
    for (auto const& kv : data) base.emplace_back(kv.key, kv.value);

    return util::timing([&] {
      dpgm_ = decltype(dpgm_)(base.begin(), base.end());

      if (base.size() > MIN_BULK_SIZE) {
        std::sort(base.begin(), base.end(),
                  [](auto const& a, auto const& b) { return a.first < b.first; });
        base.erase(
            std::unique(base.begin(), base.end(),
                        [](auto const& a, auto const& b) { return a.first == b.first; }),
            base.end());

        lipp_.bulk_load(base.data(), base.size());
        lipp_initialized_.store(true);
      }
    });
  }

  /* ---------- queries ---------- */
  uint64_t EqualityLookup(const KeyType& key, uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);

    /* try LIPP first – avoids taking PGM lock once buffer is small */
    if (lipp_initialized_.load()) {
      std::lock_guard<std::mutex> ll(lipp_mutex_);
      uint64_t v;
      if (lipp_.find(key, v)) return v;
    }

    std::lock_guard<std::mutex> dl(dpgm_mutex_);
    if (auto it = dpgm_.find(key); it != dpgm_.end()) return it->value();
    return util::NOT_FOUND;
  }

  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);

    uint64_t sum = 0;
    std::unordered_set<KeyType> seen;

    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      for (auto it = dpgm_.lower_bound(lo); it != dpgm_.end() && it->key() <= hi; ++it) {
        sum += it->value();
        seen.insert(it->key());
      }
    }

    if (lipp_initialized_.load()) {
      std::lock_guard<std::mutex> ll(lipp_mutex_);
      for (auto it = lipp_.lower_bound(lo);
           it != lipp_.end() && it->comp.data.key <= hi; ++it) {
        if (!seen.count(it->comp.data.key)) sum += it->comp.data.value;
      }
    }
    return sum;
  }

  /* ---------- insert ---------- */
  void Insert(const KeyValue<KeyType>& kv, uint32_t) {
    staging_buffer_.emplace_back(kv.key, kv.value);
    if (staging_buffer_.size() >= STAGING_CAP) flush_staging_buffer();

    uint64_t total = insert_ops_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (total % ADAPT_INTERVAL == 0) adapt_buffer_cap();
  }

  /* ---------- misc ---------- */
  bool applicable(bool, bool, bool, bool mt, const std::string&) const {
    return !mt && SearchClass::name() != "LinearAVX";
  }
  std::string name() const { return "HybridPGMLIPPAsync"; }
  std::vector<std::string> variants() const {
    return {SearchClass::name(), std::to_string(pgm_error),
            "BufferCap=" + std::to_string(buffer_cap_.load())};
  }
  std::size_t size() const {
    size_t s = dpgm_.size_in_bytes();
    if (lipp_initialized_.load()) s += lipp_.index_size();
    s += dpgm_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    return s;
  }

 private:
  /* ---------- constants ---------- */
  static constexpr size_t MIN_BULK_SIZE   = 1'000;
  static constexpr size_t STAGING_CAP     = 128;
  static constexpr size_t ADAPT_INTERVAL  = 1'000'000;
  static constexpr size_t MIN_BUF_CAP     = 50'000;
  static constexpr size_t MAX_BUF_CAP     = 5'000'000;

  /* ---------- thread-local staging ---------- */
  static thread_local std::vector<std::pair<KeyType, uint64_t>> staging_buffer_;
  void flush_staging_buffer() {
    std::vector<std::pair<KeyType, uint64_t>> buf;
    buf.swap(staging_buffer_);

    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      for (auto& kv : buf) dpgm_.insert(kv.first, kv.second);
    }

    /* flush by size */
    if (dpgm_.size() > buffer_cap_.load()) {
      auto batch_size = dpgm_.size() - buffer_cap_.load() / 2;
      auto batch      = extract_batch_from_dpgm(batch_size);
      if (!batch.empty()) {
        {
          std::lock_guard<std::mutex> ql(flush_queue_mutex_);
          flush_queue_.push(std::move(batch));
        }
        flush_cv_.notify_one();
      }
    }
  }

  /* ---------- adaptive policy ---------- */
  void adapt_buffer_cap() {
    uint64_t ins  = insert_ops_.exchange(0, std::memory_order_relaxed);
    uint64_t look = lookup_ops_.exchange(0, std::memory_order_relaxed);

    if (ins > look * 1.5) { /* insert-heavy → grow buffer */
      buffer_cap_.store(std::min(buffer_cap_.load() * 2, size_t(MAX_BUF_CAP)),
                        std::memory_order_relaxed);
    } else if (look > ins * 1.5) { /* lookup-heavy → shrink */
      buffer_cap_.store(std::max(buffer_cap_.load() / 2, size_t(MIN_BUF_CAP)),
                        std::memory_order_relaxed);
    }
  }

  /* ---------- helpers ---------- */
  std::vector<std::pair<KeyType, uint64_t>> extract_batch_from_dpgm(size_t n) {
    std::vector<std::pair<KeyType, uint64_t>> batch;
    std::lock_guard<std::mutex> dl(dpgm_mutex_);
    for (auto it = dpgm_.begin(); it != dpgm_.end() && batch.size() < n; ++it)
      batch.emplace_back(it->key(), it->value());
    for (auto& kv : batch) dpgm_.erase(kv.first);
    return batch;
  }

  void process_batch_into_lipp(std::vector<std::pair<KeyType, uint64_t>>& b) {
    std::sort(b.begin(), b.end(),
              [](auto const& a, auto const& c) { return a.first < c.first; });
    b.erase(std::unique(b.begin(), b.end(),
                        [](auto const& a, auto const& c) { return a.first == c.first; }),
            b.end());

    std::lock_guard<std::mutex> ll(lipp_mutex_);
    if (!lipp_initialized_.load()) {
      if (b.size() >= MIN_BULK_SIZE)
        lipp_.bulk_load(b.data(), b.size());
      else
        for (auto& kv : b) lipp_.insert(kv.first, kv.second);
      lipp_initialized_.store(true);
    } else {
      for (auto& kv : b) lipp_.insert(kv.first, kv.second);
    }
  }

  /* ---------- flush worker ---------- */
  void FlushThreadFunc() {
    while (true) {
      std::unique_lock<std::mutex> ql(flush_queue_mutex_);
      flush_cv_.wait(ql,
                     [this] { return !flush_queue_.empty() || stop_flag_.load(); });
      if (stop_flag_.load() && flush_queue_.empty()) break;

      auto batch = std::move(flush_queue_.front());
      flush_queue_.pop();
      ql.unlock();

      process_batch_into_lipp(batch);
    }
  }

  /* ---------- state ---------- */
  std::queue<std::vector<std::pair<KeyType, uint64_t>>> flush_queue_;
  std::mutex                                           flush_queue_mutex_;
  std::condition_variable                              flush_cv_;
  std::vector<std::thread>                             flush_threads_;
  size_t                                               num_flush_threads_;

  DynamicPGMIndex<KeyType, uint64_t, SearchClass,
                  PGMIndex<KeyType, SearchClass, pgm_error>>
      dpgm_;
  LIPP<KeyType, uint64_t> lipp_;

  mutable std::mutex dpgm_mutex_;
  mutable std::mutex lipp_mutex_;

  std::atomic<bool>   stop_flag_{false};
  std::atomic<bool>   lipp_initialized_{false};
  std::atomic<size_t> buffer_cap_{200'000};
  std::atomic<uint64_t> insert_ops_{0}, lookup_ops_{0};
};

/* instantiate thread-local buffer */
template <class K, class S, size_t E>
thread_local std::vector<std::pair<K, uint64_t>>
    HybridPGMLIPPAsync<K, S, E>::staging_buffer_;

#endif  // TLI_HYBRID_PGM_LIPP_ASYNC_H