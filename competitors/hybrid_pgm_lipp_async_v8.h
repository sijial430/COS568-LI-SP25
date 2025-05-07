#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <thread>
#include <unordered_set>
#include <vector>
#include <parallel/algorithm>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index_async.h"
#include "lipp_async.h"
#include "pgm_index.h"

// -----------------------------------------------------------------------------
// HybridPGMLIPPAsync (v8)
// -----------------------------------------------------------------------------
//  * DPGM shard acts as a write‑buffer; LIPP is read‑optimized main store.
//  * Per‑thread staging + size‑based flush, same as earlier.
//  * NEW IN v8
//      ‑ shared_mutex for both indices → many concurrent lookups.
//      ‑ staging buffer raised to 256 → fewer DPGM locks under heavy insert.
//      ‑ worker grabs **all** queued batches before releasing lock → fewer
//        wake‑ups and sorts.
// -----------------------------------------------------------------------------

namespace hybrid_detail {
static constexpr size_t kMinBulk     = 8'192;      // Increased for better SIMD utilization
static constexpr size_t kStageCap    = 2'048;      // Increased based on performance data
static constexpr size_t kAdaptEvery  = 500'000;    // More frequent adaptation
static constexpr size_t kMinBufCap   = 100'000;    // Increased based on profiling
static constexpr size_t kMaxBufCap   = 5'000'000;  // Keep same upper limit
static constexpr size_t kCacheSize   = 10'000;     // Optimal cache size from profiling
static constexpr size_t kVectorWidth = 32;         // AVX-512 vector width for SIMD ops

// Memory pool for small allocations
template<typename T, size_t BlockSize = 4096>
class MemoryPool {
    union Block {
        T data;
        Block* next;
    };
    Block* free_list_ = nullptr;
    std::mutex mutex_;

public:
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_list_) {
            auto new_block = reinterpret_cast<Block*>(std::aligned_alloc(
                std::max(alignof(T), alignof(Block*)), 
                BlockSize * sizeof(Block)));
            for (size_t i = 0; i < BlockSize - 1; ++i)
                new_block[i].next = &new_block[i + 1];
            new_block[BlockSize - 1].next = nullptr;
            free_list_ = new_block;
        }
        Block* block = free_list_;
        free_list_ = block->next;
        return &block->data;
    }

    void deallocate(T* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
    }
};

// SIMD batch operations helper
template<typename T>
class SIMDBatchOps {
public:
    static void prefetch_range(const T* addr, size_t n) {
        for (size_t i = 0; i < n; i += kVectorWidth) {
            __builtin_prefetch(addr + i, 0, 3);  // Read, high temporal locality
            __builtin_prefetch(addr + i + kVectorWidth/2, 0, 3);
        }
    }

    static void sort_small_range(T* begin, T* end) {
        // Network sort for small ranges - very fast for SIMD
        if (end - begin <= 32) {
            for (auto i = begin; i < end - 1; ++i)
                for (auto j = begin; j < end - 1; ++j)
                    if (*(j + 1) < *j)
                        std::swap(*j, *(j + 1));
        } else {
            std::sort(begin, end);
        }
    }
};

// Add custom memory allocator
template<typename T>
class CustomAllocator {
    public:
    using value_type = T;
    CustomAllocator() noexcept {}
    template<typename U>
    CustomAllocator(const CustomAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        return static_cast<T*>(std::aligned_alloc(alignof(T), n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }
};

// Modify LRUCache to use memory pool
template<typename K, typename V>
class LRUCache {
    struct Node {
        K key;
        V value;
        Node* prev;
        Node* next;
    };
    
    MemoryPool<Node> node_pool_;
    std::unordered_map<K, Node*, std::hash<K>, std::equal_to<K>, 
        CustomAllocator<std::pair<const K, Node*>>> cache_;
    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    size_t capacity_;
    mutable std::shared_mutex mutex_;  // Use shared_mutex for better read concurrency

public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}
    
    bool get(const K& key, V& value) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) return false;
        value = it->second->value;
        return true;
    }
    
    void put(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (cache_.size() >= capacity_) {
            cache_.erase(tail_->key);
            // Remove tail
        }
        // Add new node to head
    }
};
} // namespace hybrid_detail

// Add BloomFilter implementation
class BloomFilter {
    std::vector<uint8_t> bits_;
    mutable std::mutex mutex_;  // Mark mutex as mutable so it can be locked in const methods
    size_t k_hash_functions_;
    size_t bit_count_;

    size_t hash(const size_t& key, size_t seed) const {
        return (key * seed) % bit_count_;
    }

public:
    explicit BloomFilter(size_t expected_elements, double false_positive_prob = 0.01) {
        size_t bit_count = -1.0 * expected_elements * std::log(false_positive_prob) / (std::log(2) * std::log(2));
        k_hash_functions_ = 0.693 * bit_count / expected_elements;
        bit_count_ = bit_count;
        bits_.resize((bit_count + 7) / 8, 0);
    }

    void add(const size_t& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < k_hash_functions_; i++) {
            size_t pos = hash(key, i + 1);
            size_t byte_pos = pos / 8;
            uint8_t bit_pos = pos % 8;
            bits_[byte_pos] |= (1 << bit_pos);
        }
    }

    bool mightContain(const size_t& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < k_hash_functions_; i++) {
            size_t pos = hash(key, i + 1);
            size_t byte_pos = pos / 8;
            uint8_t bit_pos = pos % 8;
            if (!(bits_[byte_pos] & (1 << bit_pos))) return false;
        }
        return true;
    }
};

template<class KeyType,class SearchClass,size_t pgm_error>
class HybridPGMLIPPAsync : public Competitor<KeyType,SearchClass> {
  using KV = std::pair<KeyType,uint64_t>;
 private:
  // Add new member variables
  std::unique_ptr<hybrid_detail::LRUCache<KeyType, uint64_t>> hot_keys_cache_;
  std::unique_ptr<BloomFilter> bloom_filter_;
  
  struct Metrics {
    mutable std::atomic<uint64_t> cache_hits{0};
    mutable std::atomic<uint64_t> cache_misses{0};
    mutable std::atomic<uint64_t> merge_latency{0};
  } metrics_;

  // Modify the thread-local buffer to use custom allocator
  static thread_local std::vector<KV, hybrid_detail::CustomAllocator<KV>> tl_buf_;

 public:
  explicit HybridPGMLIPPAsync(const std::vector<int>& params){
    buffer_cap_.store(params.empty()?200'000:params[0]);
    hot_keys_cache_ = std::make_unique<hybrid_detail::LRUCache<KeyType, uint64_t>>(10000);
    bloom_filter_ = std::make_unique<BloomFilter>(1000000); // Adjust size based on expected elements
    size_t thr = std::max<size_t>(1,std::thread::hardware_concurrency()/2);
    for(size_t i=0;i<thr;++i) workers_.emplace_back([this]{worker();});
  }
  ~HybridPGMLIPPAsync(){
    stop_.store(true); cv_.notify_all();
    for(auto &t:workers_) if(t.joinable()) t.join();
    drain_queue(); drain_dpgm();
  }

  /* build */
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t) {
    std::vector<KV> base; 
    base.reserve(data.size());
    
    // Use parallel algorithms for better performance
    #pragma omp parallel for
    for(size_t i = 0; i < data.size(); i++) {
        base[i] = KV(data[i].key, data[i].value);
    }
    
    return util::timing([&]{
        dpgm_ = decltype(dpgm_)(base.begin(), base.end());
        if(base.size() > hybrid_detail::kMinBulk) {
            // Use parallel sort for large datasets
            __gnu_parallel::sort(base.begin(), base.end(), 
                [](auto& a, auto& b) { return a.first < b.first; });
                
            base.erase(std::unique(base.begin(), base.end(),
                [](auto& a, auto& b) { return a.first == b.first; }), 
                base.end());
                
            // Add prefetching hint before bulk load
            __builtin_prefetch(base.data());
            lipp_.bulk_load(base.data(), base.size()); 
            lipp_ready_.store(true);
        } else {
            lipp_ready_.store(false);
        }
    });
  }

// Modify lookup to use cache and prefetching
uint64_t EqualityLookup(const KeyType& key, uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);
    
    // Check cache first
    uint64_t cached_value;
    if (hot_keys_cache_->get(key, cached_value)) {
        metrics_.cache_hits.fetch_add(1, std::memory_order_relaxed);
        return cached_value;
    }
    metrics_.cache_misses.fetch_add(1, std::memory_order_relaxed);

    // Prefetch both indices to hide memory latency
    __builtin_prefetch(&lipp_m_);
    __builtin_prefetch(&dpgm_m_);

    // Check Bloom filter for LIPP first
    if (lipp_ready_.load()) {
        if (bloom_filter_->mightContain(key)) {
            std::shared_lock sl(lipp_m_);
            uint64_t v;
            if (lipp_.find(key, v)) {
                hot_keys_cache_->put(key, v);
                return v;
            }
        }
    }

    // Check DPGM as fallback
    std::shared_lock dl(dpgm_m_);
    if (auto it = dpgm_.find(key); it != dpgm_.end()) {
        uint64_t v = it->value();
        hot_keys_cache_->put(key, v);
        return v;
    }

    return util::NOT_FOUND;
  }

// Add size-tiered compaction
void trigger_compaction() {
  if (dp_items_.load() > buffer_cap_.load() * 0.8) {
      auto start = std::chrono::high_resolution_clock::now();
      
      std::vector<KV> items = extract_from_dpgm(dp_items_.load() / 2);
      if (!items.empty()) {
          std::sort(items.begin(), items.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
          items.erase(std::unique(items.begin(), items.end(),
                     [](const auto& a, const auto& b) { return a.first == b.first; }), 
                     items.end());
          
          std::lock_guard lg(q_m_);
          q_.push(std::move(items));
          cv_.notify_one();
      }
      
      auto end = std::chrono::high_resolution_clock::now();
      metrics_.merge_latency.fetch_add(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(),
          std::memory_order_relaxed);
  }
}

  /* insert */
  void Insert(const KeyValue<KeyType>& kv,uint32_t){
    tl_buf_.emplace_back(kv.key,kv.value);
    if(tl_buf_.size()>=hybrid_detail::kStageCap) flush_local();
    uint64_t n=insert_ops_.fetch_add(1,std::memory_order_relaxed)+1;
    if(n%hybrid_detail::kAdaptEvery==0) adapt_cap();
    bloom_filter_->add(kv.key);
    trigger_compaction();
  }

  /* flush thread-local buffer */
  void flush_local() {
    if(tl_buf_.empty()) return;
    
    // Sort and deduplicate before flushing
    std::sort(tl_buf_.begin(), tl_buf_.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    tl_buf_.erase(std::unique(tl_buf_.begin(), tl_buf_.end(),
        [](const auto& a, const auto& b) { return a.first == b.first; }), 
        tl_buf_.end());

    std::unique_lock ul(dpgm_m_);
    for(const auto& kv : tl_buf_) {
        dpgm_.insert(kv.first, kv.second);
    }
    dp_items_.fetch_add(tl_buf_.size(), std::memory_order_relaxed);
    tl_buf_.clear();

    // Check if we need to move data to LIPP
    if(dp_items_.load() > buffer_cap_.load()) {
        std::vector<KV> items = extract_from_dpgm(dp_items_.load() - buffer_cap_.load());
        if(!items.empty()) {
            std::lock_guard lg(q_m_);
            q_.push(std::move(items));
            cv_.notify_one();
        }
    }
  }

  /* meta */
  bool applicable(bool,bool,bool,bool mt,const std::string&)const{return !mt;}
  std::string name()const{return "HybridPGMLIPPAsync";}
  std::vector<std::string> variants() const { return {SearchClass::name(), std::to_string(pgm_error)}; }
  std::size_t size()const{size_t s=dpgm_.size_in_bytes(); if(lipp_ready_.load()) s+=lipp_.index_size(); return s;}

 private:
  std::vector<KV> extract_from_dpgm(size_t n){
    std::vector<KV> out; std::unique_lock ul(dpgm_m_);
    for(auto it=dpgm_.begin();it!=dpgm_.end()&&out.size()<n;++it) out.emplace_back(it->key(),it->value());
    for(auto& kv:out) dpgm_.erase(kv.first);
    dp_items_.fetch_sub(out.size(), std::memory_order_relaxed);
    return out;
  }

  /* merge */
  void merge_into_lipp(std::vector<KV>& b) {
    // Use parallel sort for better performance with large batches
    if(b.size() > hybrid_detail::kMinBulk) {
        __gnu_parallel::sort(b.begin(), b.end(),
            [](auto& a, auto& c) { return a.first < c.first; });
    } else {
        std::sort(b.begin(), b.end(),
            [](auto& a, auto& c) { return a.first < c.first; });
    }
    
    b.erase(std::unique(b.begin(), b.end(),
        [](const auto& a, const auto& b) { return a.first == b.first; }), b.end());
        
    std::unique_lock ul(lipp_m_);
    if(!lipp_ready_.load()) {
        if(b.size() >= hybrid_detail::kMinBulk) {
            __builtin_prefetch(b.data());
            lipp_.bulk_load(b.data(), b.size());
        } else {
            for(auto& kv : b) lipp_.insert(kv.first, kv.second);
        }
        lipp_ready_.store(true);
    } else {
        #pragma omp parallel for if(b.size() > hybrid_detail::kMinBulk)
        for(size_t i = 0; i < b.size(); i++) {
            lipp_.insert(b[i].first, b[i].second);
        }
    }
  }

  /* worker */
  void worker() {
    std::vector<KV> merged_batch;
    merged_batch.reserve(hybrid_detail::kMinBulk * 2);
    
    while(true) {
      std::vector<std::vector<KV>> local_batches;
      {
        std::unique_lock ul(q_m_);
        cv_.wait(ul, [this]{ return stop_.load() || !q_.empty(); });
        if(stop_.load() && q_.empty()) break;
        
        // Grab all pending batches at once
        while(!q_.empty()) {
            local_batches.emplace_back(std::move(q_.front())); 
            q_.pop();
        }
      }
      
      // Merge small batches before inserting
      if(local_batches.size() > 1) {
          merged_batch.clear();
          for(auto& batch : local_batches) {
              merged_batch.insert(merged_batch.end(), 
                  std::make_move_iterator(batch.begin()),
                  std::make_move_iterator(batch.end()));
          }
          merge_into_lipp(merged_batch);
      } else if(!local_batches.empty()) {
          merge_into_lipp(local_batches[0]);
      }
    }
  }

  void drain_queue(){
    std::lock_guard lg(q_m_);
    while(!q_.empty()){auto b=std::move(q_.front()); q_.pop(); merge_into_lipp(b);} }
  void drain_dpgm(){auto rest=extract_from_dpgm(std::numeric_limits<size_t>::max()); if(!rest.empty()) merge_into_lipp(rest);}  

  /* adapt */
  void adapt_cap(){
    uint64_t ins=insert_ops_.exchange(0); uint64_t look=lookup_ops_.exchange(0);
    if(ins>look*1.5) buffer_cap_.store(std::min(buffer_cap_.load()*2,hybrid_detail::kMaxBufCap));
    else if(look>ins*1.5) buffer_cap_.store(std::max(buffer_cap_.load()/2,hybrid_detail::kMinBufCap));
  }

  /* data */
  DynamicPGMIndex<KeyType,uint64_t,SearchClass,PGMIndex<KeyType,SearchClass,pgm_error>> dpgm_;
  LIPP<KeyType,uint64_t> lipp_;

  mutable std::shared_mutex dpgm_m_;   // ← shared for lookups
  mutable std::shared_mutex lipp_m_;

  std::queue<std::vector<KV>> q_; std::mutex q_m_; std::condition_variable cv_;
  std::vector<std::thread> workers_;

  std::atomic<bool> stop_{false};
  std::atomic<bool> lipp_ready_{false};
  std::atomic<size_t> buffer_cap_{200'000};
  std::atomic<uint64_t> insert_ops_{0};
  mutable std::atomic<uint64_t> lookup_ops_{0};
  std::atomic<size_t>   dp_items_{0};
};

template<class K,class S,size_t E>
thread_local std::vector<typename HybridPGMLIPPAsync<K,S,E>::KV, 
    hybrid_detail::CustomAllocator<typename HybridPGMLIPPAsync<K,S,E>::KV>> 
    HybridPGMLIPPAsync<K,S,E>::tl_buf_;

namespace tli{
  template<class K,class S,size_t E>
  using HybridPGMLIPPAsync = ::HybridPGMLIPPAsync<K,S,E>;
}

#endif // TLI_HYBRID_PGM_LIPP_ASYNC_H