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
    flush_threshold_ = 10000;
    if (params.size() >= 1) flush_threshold_ = params[0];
    size_t key_size = sizeof(KeyType);
    if (key_size <= 4) {
      flush_threshold_ = std::max(flush_threshold_, static_cast<size_t>(20000));
    } else if (key_size >= 16) {
      flush_threshold_ = std::min(flush_threshold_, static_cast<size_t>(5000));
    }
    stop_flag_ = false;
    flush_thread_ = std::thread([this] { this->FlushThreadFunc(); });
  }

  ~HybridPGMLIPPAsync() {
    stop_flag_ = true;
    flush_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();
    // Final flush for any remaining data
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (!active_buffer_.empty()) {
      flush_buffer_.insert(active_buffer_.begin(), active_buffer_.end());
      active_buffer_.clear();
    }
    if (!flush_buffer_.empty()) {
      FlushBufferUnlocked();
    }
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
    std::vector<std::pair<KeyType, uint64_t>> base_data;
    base_data.reserve(data.size());
    for (const auto& item : data) {
      base_data.emplace_back(item.key, item.value);
    }

    // Adjust flush threshold based on data size
    if (data.size() > 1000000) {
      // For large datasets, use larger batches
      flush_threshold_ = std::max(flush_threshold_, data.size() / 100);
    }

    // Build both indexes
    return util::timing([&] {
      // Build PGM index
      pgm_ = decltype(pgm_)(base_data.begin(), base_data.end());
      
      // Build LIPP index if enough data
      if (base_data.size() > MIN_BULK_SIZE) {
        lipp_.bulk_load(base_data.data(), base_data.size());
        lipp_initialized_ = true;
      } else {
        lipp_initialized_ = false;
      }
    });
  }

  size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
    // First check both buffers (fastest)
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      auto it = active_buffer_.find(key);
      if (it != active_buffer_.end()) return it->second;
      it = flush_buffer_.find(key);
      if (it != flush_buffer_.end()) return it->second;
    }
    // Then check LIPP if initialized (second fastest)
    if (lipp_initialized_) {
      uint64_t value;
      if (lipp_.find(key, value)) {
        return value;
      }
    }
    // Finally check PGM
    auto pgm_it = pgm_.find(key);
    return (pgm_it != pgm_.end()) ? pgm_it->value() : util::NOT_FOUND;
  }

  void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
    pgm_.insert(data.key, data.value);
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      active_buffer_[data.key] = data.value;
      if (++insert_count_ >= flush_threshold_) {
        active_buffer_.swap(flush_buffer_);
        insert_count_ = 0;
        flush_cv_.notify_one();
      }
    }
  }

  uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key, uint32_t thread_id) const {
    uint64_t result = 0;
    std::unordered_map<KeyType, bool> seen_keys;
    // Check both buffers first
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      for (const auto& [key, value] : active_buffer_) {
        if (key >= lower_key && key <= upper_key) {
          result += value;
          seen_keys[key] = true;
        }
      }
      for (const auto& [key, value] : flush_buffer_) {
        if (key >= lower_key && key <= upper_key) {
          result += value;
          seen_keys[key] = true;
        }
      }
    }
    // Then check LIPP if initialized
    if (lipp_initialized_) {
      auto it = lipp_.lower_bound(lower_key);
      while (it != lipp_.end() && it->comp.data.key <= upper_key) {
        if (seen_keys.find(it->comp.data.key) == seen_keys.end()) {
          result += it->comp.data.value;
          seen_keys[it->comp.data.key] = true;
        }
        ++it;
      }
    }
    // Finally check PGM
    auto it = pgm_.lower_bound(lower_key);
    while (it != pgm_.end() && it->key() <= upper_key) {
      if (seen_keys.find(it->key()) == seen_keys.end()) {
        result += it->value();
      }
      ++it;
    }
    return result;
  }

  bool applicable(bool unique, bool range_query, bool insert, bool multithread, const std::string& ops_filename) const {
    std::string name = SearchClass::name();
    return name != "LinearAVX" && !multithread;
  }

  std::string name() const { return "HybridPGMLIPPAsync"; }

  std::vector<std::string> variants() const { 
    std::vector<std::string> vec;
    vec.push_back(SearchClass::name());
    vec.push_back(std::to_string(pgm_error));
    vec.push_back(std::string("FlushThreshold=") + std::to_string(flush_threshold_));
    return vec;
  }

  std::size_t size() const { 
    size_t total = pgm_.size_in_bytes();
    if (lipp_initialized_) total += lipp_.index_size();
    // Estimate buffer size
    total += active_buffer_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    total += flush_buffer_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    return total;
  }

private:
  static constexpr size_t MIN_BULK_SIZE = 1000;

  void FlushThreadFunc() {
    while (!stop_flag_) {
      std::unique_lock<std::mutex> lock(buffer_mutex_);
      flush_cv_.wait(lock, [this] { return !flush_buffer_.empty() || stop_flag_; });
      if (!flush_buffer_.empty()) {
        FlushBufferUnlocked();
      }
    }
  }

  void FlushBufferUnlocked() {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (!lipp_initialized_ && flush_buffer_.size() >= MIN_BULK_SIZE) {
      std::vector<std::pair<KeyType, uint64_t>> data;
      data.reserve(flush_buffer_.size());
      for (const auto& [key, value] : flush_buffer_) {
        data.emplace_back(key, value);
      }
      std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      lipp_.bulk_load(data.data(), data.size());
      lipp_initialized_ = true;
    } else if (lipp_initialized_) {
      for (const auto& [key, value] : flush_buffer_) {
        lipp_.insert(key, value);
      }
    }
    flush_buffer_.clear();
    auto end_time = std::chrono::high_resolution_clock::now();
    buffer_flush_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    // Dynamic adjustment of flush threshold based on performance
    if (buffer_flush_time_ > 0) {
      if (buffer_flush_time_ > 10000000) {
        flush_threshold_ = std::max(static_cast<size_t>(1000), flush_threshold_ / 2);
      } else if (buffer_flush_time_ < 1000000) {
        flush_threshold_ = std::min(static_cast<size_t>(50000), flush_threshold_ * 2);
      }
    }
  }

  // Configuration
  size_t flush_threshold_ = 10000;
  size_t insert_count_ = 0;
  uint64_t buffer_flush_time_ = 0;

  // Data structures
  DynamicPGMIndex<KeyType, uint64_t, SearchClass, PGMIndex<KeyType, SearchClass, pgm_error>> pgm_;
  LIPP<KeyType, uint64_t> lipp_;
  bool lipp_initialized_ = false;
  std::unordered_map<KeyType, uint64_t> active_buffer_;
  std::unordered_map<KeyType, uint64_t> flush_buffer_;
  // std::mutex buffer_mutex_;
  mutable std::mutex buffer_mutex_;   // accessed from const methods
  std::condition_variable flush_cv_;
  std::thread flush_thread_;
  std::atomic<bool> stop_flag_;
};

#endif  // TLI_HYBRID_PGM_LIPP_ASYNC_H