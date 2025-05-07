#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <unordered_set>

#include "../util.h"
#include "base.h"
#include "pgm_index.h"
#include "dynamic_pgm_index_async.h"
#include "lipp_async.h"
#include "../searches/linear_search.h"
#include "../searches/branching_binary_search.h"
#include "../searches/interpolation_search.h"
#include "../searches/exponential_search.h"

template <class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPPAsync : public Competitor<KeyType, SearchClass> {
public:
  HybridPGMLIPPAsync(const std::vector<int>& params) {
    // … unchanged …
    stop_flag_.store(false);
    insert_count_.store(0);
    for (size_t i = 0; i < num_flush_threads_; ++i)
      flush_threads_.emplace_back([this]{ FlushThreadFunc(); });
  }

  ~HybridPGMLIPPAsync() {
    // 1) stop flush threads
    stop_flag_.store(true);
    flush_cv_.notify_all();
    for (auto& t : flush_threads_) if (t.joinable()) t.join();

    // 2) drain any pending flush‐queue batches
    while (true) {
      std::vector<std::pair<KeyType,uint64_t>> batch;
      {
        std::lock_guard<std::mutex> ql(flush_queue_mutex_);
        if (flush_queue_.empty()) break;
        batch = std::move(flush_queue_.front());
        flush_queue_.pop();
      }
      process_batch_into_lipp(batch);
    }

    // 3) final flush from DPGM
    std::vector<std::pair<KeyType,uint64_t>> final_batch;
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      for (auto it = dpgm_.begin(); it != dpgm_.end(); ++it)
        final_batch.emplace_back(it->key(), it->value());
      for (auto& kv : final_batch)
        dpgm_.erase(kv.first);
    }
    if (!final_batch.empty())
      process_batch_into_lipp(final_batch);
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
    std::vector<std::pair<KeyType,uint64_t>> base_data;
    base_data.reserve(data.size());
    for (auto const& item : data)
      base_data.emplace_back(item.key, item.value);

    if (data.size() > 1'000'000)
      flush_threshold_ = std::max(flush_threshold_, data.size()/100);

    return util::timing([&]{
      dpgm_ = decltype(dpgm_)(base_data.begin(), base_data.end());

      if (base_data.size() > MIN_BULK_SIZE) {
        // **SORT + DEDUPE before first bulk_load**
        std::sort(base_data.begin(), base_data.end(),
                  [](auto const&a, auto const&b){ return a.first < b.first; });
        base_data.erase(std::unique(base_data.begin(), base_data.end(),
                                    [](auto const&a, auto const&b){ return a.first==b.first; }),
                        base_data.end());

        lipp_.bulk_load(base_data.data(), base_data.size());
        lipp_initialized_.store(true);
      } else {
        lipp_initialized_.store(false);
      }
    });
  }

  uint64_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      auto it = dpgm_.find(key);
      if (it != dpgm_.end()) return it->value();
    }
    if (lipp_initialized_.load()) {
      std::lock_guard<std::mutex> ll(lipp_mutex_);
      uint64_t v;
      if (lipp_.find(key, v)) return v;
    }
    return util::NOT_FOUND;
  }

  void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      dpgm_.insert(data.key, data.value);
    }
    size_t cur = insert_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (cur >= flush_threshold_) {
      auto batch = extract_batch_from_dpgm(flush_threshold_);
      if (!batch.empty()) {
        {
          std::lock_guard<std::mutex> ql(flush_queue_mutex_);
          flush_queue_.push(std::move(batch));
        }
        insert_count_.fetch_sub(batch.size(), std::memory_order_relaxed);
        flush_cv_.notify_one();
      }
    }
  }

  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t) const {
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
      for (auto it = lipp_.lower_bound(lo); it != lipp_.end() && it->comp.data.key <= hi; ++it) {
        if (!seen.count(it->comp.data.key))
          sum += it->comp.data.value;
      }
    }
    return sum;
  }

  bool applicable(bool unique, bool range_q, bool insert, bool mt, const std::string&) const {
    return SearchClass::name() != "LinearAVX" && !mt;
  }

  std::string name() const { return "HybridPGMLIPPAsync"; }
  std::vector<std::string> variants() const {
    return { SearchClass::name(),
             std::to_string(pgm_error),
             "FlushThreshold=" + std::to_string(flush_threshold_) };
  }
  std::size_t size() const {
    size_t s = dpgm_.size_in_bytes();
    if (lipp_initialized_.load()) s += lipp_.index_size();
    s += dpgm_.size()*(sizeof(KeyType)+sizeof(uint64_t));
    return s;
  }

private:
  static constexpr size_t MIN_BULK_SIZE = 1000;

  // the shared flush queue
  std::queue<std::vector<std::pair<KeyType,uint64_t>>> flush_queue_;
  std::mutex flush_queue_mutex_;
  std::condition_variable flush_cv_;

  // flush worker threads
  std::vector<std::thread> flush_threads_;
  size_t num_flush_threads_ = std::max<size_t>(1, std::thread::hardware_concurrency()/2);

  // indexes + locks
  DynamicPGMIndex<KeyType,uint64_t,SearchClass,PGMIndex<KeyType,SearchClass,pgm_error>> dpgm_;
  LIPP<KeyType,uint64_t> lipp_;
  mutable std::mutex dpgm_mutex_, lipp_mutex_;

  // flags & counters
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> lipp_initialized_{false};
  std::atomic<size_t> insert_count_{0};
  size_t flush_threshold_{10000};

  // extract from DPGM under lock
  std::vector<std::pair<KeyType,uint64_t>> extract_batch_from_dpgm(size_t n) {
    std::vector<std::pair<KeyType,uint64_t>> b;
    std::lock_guard<std::mutex> lk(dpgm_mutex_);
    for (auto it = dpgm_.begin(); it != dpgm_.end() && b.size()<n; ++it)
      b.emplace_back(it->key(), it->value());
    for (auto& kv : b) dpgm_.erase(kv.first);
    return b;
  }

  // unified “sort+dedupe, then bulk or per-key insert” logic
  void process_batch_into_lipp(std::vector<std::pair<KeyType,uint64_t>>& batch) {
    // always sort & unique first
    std::sort(batch.begin(), batch.end(),
              [](auto const&a, auto const&b){ return a.first < b.first; });
    batch.erase(std::unique(batch.begin(), batch.end(),
                            [](auto const&a, auto const&b){ return a.first==b.first; }),
                batch.end());

    std::lock_guard<std::mutex> lk(lipp_mutex_);
    if (!lipp_initialized_.load()) {
      if (batch.size() >= MIN_BULK_SIZE) {
        lipp_.bulk_load(batch.data(), batch.size());
      } else {
        for (auto& kv : batch)
          lipp_.insert(kv.first, kv.second);
      }
      lipp_initialized_.store(true);
    } else {
      for (auto& kv : batch)
        lipp_.insert(kv.first, kv.second);
    }
  }

  // flush‐thread main loop
  void FlushThreadFunc() {
    while (true) {
      std::unique_lock<std::mutex> ql(flush_queue_mutex_);
      flush_cv_.wait(ql, [this]{ return !flush_queue_.empty() || stop_flag_.load(); });
      if (stop_flag_.load() && flush_queue_.empty()) break;

      auto batch = std::move(flush_queue_.front());
      flush_queue_.pop();
      ql.unlock();
      process_batch_into_lipp(batch);
    }
  }
};

#endif  // TLI_HYBRID_PGM_LIPP_ASYNC_H
