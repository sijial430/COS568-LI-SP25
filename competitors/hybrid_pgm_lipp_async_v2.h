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
#include <memory>
#include <limits>

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
    
    // Adjust flush threshold based on key size
    size_t key_size = sizeof(KeyType);
    if (key_size <= 4) {
      flush_threshold_ = std::max(flush_threshold_, static_cast<size_t>(20000));
    } else if (key_size >= 16) {
      flush_threshold_ = std::min(flush_threshold_, static_cast<size_t>(5000));
    }
    
    // Start background flush thread
    stop_flag_ = false;
    flush_thread_ = std::thread([this] { this->FlushThreadFunc(); });
  }

  ~HybridPGMLIPPAsync() {
    // Signal stop and join flush thread
    stop_flag_ = true;
    flush_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();
    
    // Final flush for any remaining data
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    // First move any active buffer data to flush buffer
    if (!active_buffer_.empty()) {
      flush_buffer_.insert(active_buffer_.begin(), active_buffer_.end());
      active_buffer_.clear();
    }
    // Then flush any remaining keys
    if (!flush_buffer_.empty()) {
      FlushBufferUnlocked();
    }
    
    // Final flush of keys from PGM to LIPP if needed
    if (lipp_initialized_ && perform_pgm_flush_) {
      FlushPGMToLIPPUnlocked();
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
    
    // Adjust PGM flush threshold - default to 5% of dataset size
    pgm_flush_threshold_ = data.size() / 20;
    pgm_flush_threshold_ = std::max(pgm_flush_threshold_, static_cast<size_t>(10000));

    // Build both indexes
    return util::timing([&] {
      // Build PGM index
      pgm_ = decltype(pgm_)(base_data.begin(), base_data.end());
      
      // Build LIPP index if enough data
      if (base_data.size() > MIN_BULK_SIZE) {
        lipp_.bulk_load(base_data.data(), base_data.size());
        lipp_initialized_ = true;
        
        // Store initial size to track growth for PGM flush threshold
        initial_pgm_size_ = pgm_.size_in_bytes();
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
      
      // Check if we need to schedule a buffer flush
      if (++insert_count_ >= flush_threshold_) {
        active_buffer_.swap(flush_buffer_);
        insert_count_ = 0;
        flush_cv_.notify_one();
      }
      
      // Check if we need to schedule a PGM flush
      size_t pgm_current_size = pgm_.size_in_bytes();
      if (lipp_initialized_ && 
          !perform_pgm_flush_ && 
          (pgm_current_size - initial_pgm_size_ > pgm_flush_threshold_)) {
        perform_pgm_flush_ = true;
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
        if (key >= lower_key && key <= upper_key && seen_keys.find(key) == seen_keys.end()) {
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
    vec.push_back(std::string("PGMFlushThreshold=") + std::to_string(pgm_flush_threshold_));
    return vec;
  }

  std::size_t size() const { 
    size_t total = pgm_.size_in_bytes();
    if (lipp_initialized_) total += lipp_.index_size();
    
    // Estimate buffer size
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    total += active_buffer_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    total += flush_buffer_.size() * (sizeof(KeyType) + sizeof(uint64_t));
    
    return total;
  }

private:
  static constexpr size_t MIN_BULK_SIZE = 1000;
  static constexpr size_t BATCH_SIZE = 1000; // For PGM to LIPP batch operations

  void FlushThreadFunc() {
    while (!stop_flag_) {
      std::unique_lock<std::mutex> lock(buffer_mutex_);
      
      // Wait until there's work to do or we're stopping
      flush_cv_.wait(lock, [this] { 
        return !flush_buffer_.empty() || perform_pgm_flush_ || stop_flag_; 
      });
      
      // First handle buffer flush (higher priority)
      if (!flush_buffer_.empty()) {
        FlushBufferUnlocked();
      }
      
      // Then handle PGM flush if needed
      if (perform_pgm_flush_ && lipp_initialized_) {
        FlushPGMToLIPPUnlocked();
        perform_pgm_flush_ = false;
        initial_pgm_size_ = pgm_.size_in_bytes();
      }
    }
  }

  void FlushBufferUnlocked() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize LIPP with bulk load if not initialized yet and buffer is large enough
    if (!lipp_initialized_ && flush_buffer_.size() >= MIN_BULK_SIZE) {
      std::vector<std::pair<KeyType, uint64_t>> data;
      data.reserve(flush_buffer_.size());
      
      for (const auto& [key, value] : flush_buffer_) {
        data.emplace_back(key, value);
      }
      
      // Sort data for optimal bulk loading
      std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) { 
        return a.first < b.first; 
      });
      
      lipp_.bulk_load(data.data(), data.size());
      lipp_initialized_ = true;
    } 
    // If LIPP is initialized, use batch inserts when possible
    else if (lipp_initialized_) {
      // For large buffers, use sorted batch insertion
      if (flush_buffer_.size() >= MIN_BULK_SIZE) {
        std::vector<std::pair<KeyType, uint64_t>> data;
        data.reserve(flush_buffer_.size());
        
        for (const auto& [key, value] : flush_buffer_) {
          data.emplace_back(key, value);
        }
        
        // Sort data for optimal insertion
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) { 
          return a.first < b.first; 
        });
        
        // Insert in batches to avoid long blocking operations
        for (size_t i = 0; i < data.size(); i += BATCH_SIZE) {
          size_t batch_end = std::min(i + BATCH_SIZE, data.size());
          for (size_t j = i; j < batch_end; j++) {
            lipp_.insert(data[j].first, data[j].second);
          }
          
          // Brief yield to allow other operations
          if (batch_end < data.size()) {
            buffer_mutex_.unlock();
            std::this_thread::yield();
            buffer_mutex_.lock();
          }
        }
      } 
      // For small buffers, insert directly
      else {
        for (const auto& [key, value] : flush_buffer_) {
          lipp_.insert(key, value);
        }
      }
    }
    
    flush_buffer_.clear();
    
    // Measure and adjust the flush threshold based on performance
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t flush_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time).count();
    buffer_flush_time_ = flush_time;
    
    // Dynamic adjustment of flush threshold based on performance
    if (flush_time > 0) {
      if (flush_time > 10000000) { // 10ms
        // Reduce threshold if flushing takes too long
        flush_threshold_ = std::max(static_cast<size_t>(1000), flush_threshold_ / 2);
      } else if (flush_time < 1000000) { // 1ms
        // Increase threshold if flushing is very fast
        flush_threshold_ = std::min(static_cast<size_t>(50000), flush_threshold_ * 2);
      }
    }
  }

  void FlushPGMToLIPPUnlocked() {
    if (!lipp_initialized_) return;
    
    // We take a snapshot of the PGM index and move entries to LIPP
    // This is a simplified approach - for a production system, we would
    // need more complex logic to handle concurrent modifications
    
    std::vector<std::pair<KeyType, uint64_t>> pgm_data;
    std::unordered_map<KeyType, bool> buffer_keys;
    
    // First collect keys from buffers to avoid duplicates
    for (const auto& [key, _] : active_buffer_) {
      buffer_keys[key] = true;
    }
    for (const auto& [key, _] : flush_buffer_) {
      buffer_keys[key] = true;
    }
    
    // Instead of using iterators directly, use a different approach to extract keys from PGM
    // This avoids iterator compatibility issues with DynamicPGMIndex
    
    // Option 1: Use a range-based approach with lower/upper bounds
    KeyType min_key = std::numeric_limits<KeyType>::min();
    KeyType max_key = std::numeric_limits<KeyType>::max();
    KeyType current_key = min_key;
    
    while (current_key < max_key) {
      auto it = pgm_.lower_bound(current_key);
      if (it == pgm_.end()) break;
      
      KeyType key = it->key();
      uint64_t value = it->value();
      
      // Move to next key
      if (key <= current_key) {
        // If we didn't advance, avoid infinite loop
        current_key = key + 1;
        // For non-integer types, we need a different approach
        if (current_key <= key) break;
      } else {
        current_key = key + 1;
      }
      
      // Add key if it's not in the buffers
      if (buffer_keys.find(key) == buffer_keys.end()) {
        pgm_data.emplace_back(key, value);
      }
      
      // Periodically yield to avoid blocking for too long
      if (pgm_data.size() % BATCH_SIZE == 0) {
        buffer_mutex_.unlock();
        std::this_thread::yield();
        buffer_mutex_.lock();
      }
    }
    
    // Sort data for optimal insertion
    std::sort(pgm_data.begin(), pgm_data.end(), [](const auto& a, const auto& b) { 
      return a.first < b.first; 
    });
    
    // Insert data into LIPP in batches
    for (size_t i = 0; i < pgm_data.size(); i += BATCH_SIZE) {
      size_t batch_end = std::min(i + BATCH_SIZE, pgm_data.size());
      
      for (size_t j = i; j < batch_end; j++) {
        lipp_.insert(pgm_data[j].first, pgm_data[j].second);
      }
      
      // Brief yield to allow other operations
      if (batch_end < pgm_data.size()) {
        buffer_mutex_.unlock();
        std::this_thread::yield();
        buffer_mutex_.lock();
      }
    }
  }

  // Configuration
  size_t flush_threshold_ = 10000;           // Buffer flush threshold
  size_t pgm_flush_threshold_ = 10000;       // PGM to LIPP flush threshold
  size_t insert_count_ = 0;                  // Counter for inserts since last flush
  uint64_t buffer_flush_time_ = 0;           // Time taken for last buffer flush
  size_t initial_pgm_size_ = 0;              // Initial PGM size after build
  bool perform_pgm_flush_ = false;           // Flag to indicate PGM flush is needed

  // Data structures
  DynamicPGMIndex<KeyType, uint64_t, SearchClass, PGMIndex<KeyType, SearchClass, pgm_error>> pgm_;
  LIPP<KeyType, uint64_t> lipp_;
  bool lipp_initialized_ = false;
  std::unordered_map<KeyType, uint64_t> active_buffer_;  // Buffer for active inserts
  std::unordered_map<KeyType, uint64_t> flush_buffer_;   // Buffer being flushed
  
  // Synchronization
  mutable std::mutex buffer_mutex_;            // Protects buffers and flags
  std::condition_variable flush_cv_;           // For signaling flush thread
  std::thread flush_thread_;                   // Background flush thread
  std::atomic<bool> stop_flag_;                // Flag to stop background thread
};

#endif  // TLI_HYBRID_PGM_LIPP_ASYNC_H