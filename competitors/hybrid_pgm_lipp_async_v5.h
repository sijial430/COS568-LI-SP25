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
#include <shared_mutex>
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
    size_t key_size = sizeof(KeyType);
    flush_threshold_.store(params.size()>=1 ? params[0] : 10'000);
    if (key_size <= 4)
      flush_threshold_.store(std::max(flush_threshold_.load(), size_t(20'000)));
    else if (key_size >= 16)
      flush_threshold_.store(std::min(flush_threshold_.load(), size_t(5'000)));

    stop_flag_.store(false);
    insert_ops_.store(0);
    lookup_ops_.store(0);

    for (size_t i = 0; i < num_flush_threads_; ++i)
      flush_threads_.emplace_back([this]{ FlushThreadFunc(); });
  }

  ~HybridPGMLIPPAsync() {
    // 1) stop flush threads
    stop_flag_.store(true);
    flush_cv_.notify_all();
    for (auto& t : flush_threads_) if (t.joinable()) t.join();

    // 2) drain queue
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

    // 3) final DPGM flush
    std::vector<std::pair<KeyType,uint64_t>> final_batch;
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      final_batch.reserve(dpgm_.size());
      for (auto it=dpgm_.begin(); it!=dpgm_.end(); ++it)
        final_batch.emplace_back(it->key(), it->value());
      for (auto& kv: final_batch)
        dpgm_.erase(kv.first);
    }
    if (!final_batch.empty())
      process_batch_into_lipp(final_batch);
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t) {
    std::vector<std::pair<KeyType,uint64_t>> base;
    base.reserve(data.size());
    for (auto const& kv : data) base.emplace_back(kv.key, kv.value);

    if (data.size()>1'000'000)
      flush_threshold_.store(std::max(flush_threshold_.load(), data.size()/100));

    return util::timing([&]{
      dpgm_ = decltype(dpgm_)(base.begin(), base.end());
      if (base.size()>MIN_BULK_SIZE) {
        std::sort(base.begin(), base.end(),
                  [](auto const&a, auto const&b){ return a.first<b.first; });
        base.erase(std::unique(base.begin(),base.end(),
                               [](auto const&a,auto const&b){return a.first==b.first;}),
                   base.end());
        lipp_.bulk_load(base.data(), base.size());
        lipp_initialized_.store(true);
      } else {
        lipp_initialized_.store(false);
      }
    });
  }

  uint64_t EqualityLookup(const KeyType& key, uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      if (auto it = dpgm_.find(key); it!=dpgm_.end())
        return it->value();
    }
    if (lipp_initialized_.load()) {
      std::lock_guard<std::mutex> ll(lipp_mutex_);
      uint64_t v;
      if (lipp_.find(key,v)) return v;
    }
    return util::NOT_FOUND;
  }

  void Insert(const KeyValue<KeyType>& data, uint32_t) {
    // stage locally
    staging_buffer_.emplace_back(data.key,data.value);
    if (staging_buffer_.size()>=STAGING_CAP) {
      flush_staging();
    }
    auto total = insert_ops_.fetch_add(1, std::memory_order_relaxed)+1;
    if (total % ADAPT_INTERVAL == 0)
      adapt_threshold();
  }

  uint64_t RangeQuery(const KeyType& lo,const KeyType& hi,uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);
    uint64_t sum=0;
    std::unordered_set<KeyType> seen;
    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      for (auto it=dpgm_.lower_bound(lo); it!=dpgm_.end() && it->key()<=hi; ++it) {
        sum += it->value();
        seen.insert(it->key());
      }
    }
    if (lipp_initialized_.load()) {
      std::lock_guard<std::mutex> ll(lipp_mutex_);
      for (auto it=lipp_.lower_bound(lo); it!=lipp_.end() && it->comp.data.key<=hi; ++it){
        if (!seen.count(it->comp.data.key))
          sum += it->comp.data.value;
      }
    }
    return sum;
  }

  bool applicable(bool u,bool rq,bool i,bool mt,const std::string&) const {
    return SearchClass::name()!="LinearAVX" && !mt;
  }

  std::string name() const { return "HybridPGMLIPPAsync"; }
  std::vector<std::string> variants() const {
    return {SearchClass::name(),
            std::to_string(pgm_error),
            "FlushThreshold="+std::to_string(flush_threshold_.load())};
  }
  std::size_t size() const {
    size_t s = dpgm_.size_in_bytes();
    if (lipp_initialized_.load()) s+=lipp_.index_size();
    s += dpgm_.size()*(sizeof(KeyType)+sizeof(uint64_t));
    return s;
  }

private:
  static constexpr size_t MIN_BULK_SIZE   = 1'000;
  static constexpr size_t STAGING_CAP     = 128;
  static constexpr size_t ADAPT_INTERVAL  = 1'000'000;
  static constexpr size_t MIN_FLUSH_THR   = 1'000;
  static constexpr size_t MAX_FLUSH_THR   =10'000'000;

  // staging per thread
  static thread_local std::vector<std::pair<KeyType,uint64_t>> staging_buffer_;

  void flush_staging() {
    std::vector<std::pair<KeyType,uint64_t>> buf;
    buf.swap(staging_buffer_);

    {
      std::lock_guard<std::mutex> dl(dpgm_mutex_);
      for (auto& kv: buf) dpgm_.insert(kv.first, kv.second);
    }

    size_t cur = insert_count_.fetch_add(buf.size(), std::memory_order_relaxed) + buf.size();
    if (cur >= flush_threshold_.load()) {
      auto batch = extract_batch_from_dpgm(flush_threshold_.load());
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

  void adapt_threshold() {
    uint64_t ins = insert_ops_.exchange(0, std::memory_order_relaxed);
    uint64_t look= lookup_ops_.exchange(0,std::memory_order_relaxed);
    if (ins>look*1.5) {
      flush_threshold_.store(
        std::min(flush_threshold_.load()*2, MAX_FLUSH_THR),
        std::memory_order_relaxed);
    } else if (look>ins*1.5) {
      flush_threshold_.store(
        std::max(flush_threshold_.load()/2, MIN_FLUSH_THR),
        std::memory_order_relaxed);
    }
  }

  // queue + threads
  std::queue<std::vector<std::pair<KeyType,uint64_t>>> flush_queue_;
  std::mutex                                 flush_queue_mutex_;
  std::condition_variable                    flush_cv_;
  std::vector<std::thread>                   flush_threads_;
  size_t                                     num_flush_threads_ = std::max<size_t>(1,std::thread::hardware_concurrency()/2);

  // indexes + locks
  DynamicPGMIndex<KeyType,uint64_t,SearchClass,PGMIndex<KeyType,SearchClass,pgm_error>> dpgm_;
  LIPP<KeyType,uint64_t>                                        lipp_;
  mutable std::mutex                                            dpgm_mutex_;
  mutable std::mutex                                            lipp_mutex_;

  // flags & counters
  std::atomic<bool>            stop_flag_{false};
  std::atomic<bool>            lipp_initialized_{false};
  std::atomic<size_t>          insert_count_{0};
  std::atomic<uint64_t>        insert_ops_{0}, lookup_ops_{0};
  std::atomic<size_t>          flush_threshold_{10'000};

  std::vector<std::pair<KeyType,uint64_t>> extract_batch_from_dpgm(size_t n) {
    std::vector<std::pair<KeyType,uint64_t>> b;
    std::lock_guard<std::mutex> dl(dpgm_mutex_);
    for (auto it=dpgm_.begin(); it!=dpgm_.end() && b.size()<n; ++it)
      b.emplace_back(it->key(), it->value());
    for (auto& kv: b) dpgm_.erase(kv.first);
    return b;
  }

  void process_batch_into_lipp(std::vector<std::pair<KeyType,uint64_t>>& batch) {
    std::sort(batch.begin(),batch.end(),
              [](auto const&a,auto const&b){return a.first<b.first;});
    batch.erase(std::unique(batch.begin(),batch.end(),
                            [](auto const&a,auto const&b){return a.first==b.first;}),
                batch.end());
    std::lock_guard<std::mutex> ll(lipp_mutex_);
    if (!lipp_initialized_.load()) {
      if (batch.size()>=MIN_BULK_SIZE)
        lipp_.bulk_load(batch.data(), batch.size());
      else
        for (auto& kv: batch) lipp_.insert(kv.first, kv.second);
      lipp_initialized_.store(true);
    } else {
      for (auto& kv: batch) lipp_.insert(kv.first, kv.second);
    }
  }

  void FlushThreadFunc() {
    while (true) {
      std::unique_lock<std::mutex> ql(flush_queue_mutex_);
      flush_cv_.wait(ql, [this]{return !flush_queue_.empty()||stop_flag_.load();});
      if (stop_flag_.load() && flush_queue_.empty()) break;
      auto batch = std::move(flush_queue_.front());
      flush_queue_.pop();
      ql.unlock();
      process_batch_into_lipp(batch);
    }
  }
};

template <class K,class S,size_t E>
thread_local std::vector<std::pair<K,uint64_t>> HybridPGMLIPPAsync<K,S,E>::staging_buffer_;

#endif // TLI_HYBRID_PGM_LIPP_ASYNC_H