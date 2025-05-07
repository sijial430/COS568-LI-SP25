#ifndef TLI_HYBRID_PGM_LIPP_ASYNC_H
#define TLI_HYBRID_PGM_LIPP_ASYNC_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <thread>
#include <vector>
#include <parallel/algorithm>
#include <unordered_set>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include "pgm_index.h"

namespace hybrid_detail {
static constexpr size_t kMinBulk = 8'192;      // For batch efficiency
static constexpr size_t kStageCap = 2'048;     // Local buffer size
static constexpr size_t kAdaptEvery = 1'000'000; // Adaptation frequency
static constexpr size_t kMinBufCap = 50'000;  // Min buffer capacity
static constexpr size_t kMaxBufCap = 5'000'000;// Max buffer capacity
} // namespace hybrid_detail

// namespace hybrid_detail {
// static constexpr size_t kMinBulk     = 1'000;
// static constexpr size_t kStageCap   = 256;        // ↑ from 128
// static constexpr size_t kAdaptEvery = 1'000'000;
// static constexpr size_t kMinBufCap  = 50'000;
// static constexpr size_t kMaxBufCap  = 5'000'000;
// } // namespace hybrid_detail

template<class KeyType,class SearchClass,size_t pgm_error>
class HybridPGMLIPPAsync : public Competitor<KeyType,SearchClass> {
  using KV = std::pair<KeyType,uint64_t>;
 public:
  explicit HybridPGMLIPPAsync(const std::vector<int>& params){
    buffer_cap_.store(params.empty()?200'000:params[0]);
    size_t thr = std::max<size_t>(1,std::thread::hardware_concurrency()/2);
    for(size_t i=0;i<thr;++i) workers_.emplace_back([this]{worker();});
  }

  ~HybridPGMLIPPAsync(){
    stop_.store(true); cv_.notify_all();
    for(auto &t:workers_) if(t.joinable()) t.join();
    drain_queue(); drain_dpgm();
  }

  /* build */
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data,size_t){
    std::vector<KV> base; base.reserve(data.size());
    for(auto const& kv:data) base.emplace_back(kv.key,kv.value);
    return util::timing([&]{
      dpgm_ = decltype(dpgm_)(base.begin(),base.end());
      if(base.size()>hybrid_detail::kMinBulk){
        std::sort(base.begin(),base.end(),[](auto&a,auto&b){return a.first<b.first;});
        base.erase(std::unique(base.begin(),base.end(),[](auto&a,auto&b){return a.first==b.first;}),base.end());
        lipp_.Build(data, 1); lipp_ready_.store(true);
      }else lipp_ready_.store(false);
    });
  }

  /* lookup */
  uint64_t EqualityLookup(const KeyType& key,uint32_t)const{
    lookup_ops_.fetch_add(1,std::memory_order_relaxed);
    if(lipp_ready_.load()){
      std::shared_lock sl(lipp_m_);
      auto v = lipp_.EqualityLookup(key,0); if(v!=util::NOT_FOUND) return v;
    }
    std::shared_lock dl(dpgm_m_);
    if(auto it=dpgm_.find(key); it!=dpgm_.end()) return it->value();
    return util::NOT_FOUND;
  }
  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t) const {
    lookup_ops_.fetch_add(1, std::memory_order_relaxed);

    uint64_t sum = 0;
    std::unordered_set<KeyType> seen;
    /* scan write‑buffer */
    {
      std::shared_lock dl(dpgm_m_);
      for (auto it = dpgm_.lower_bound(lo); it != dpgm_.end() && it->key() <= hi; ++it) {
        sum += it->value();
        seen.insert(it->key());
      }
    }
    /* add from LIPPAsync, skipping duplicates we already saw */
    if (lipp_ready_.load()) {
      std::shared_lock sl(lipp_m_);
      sum += lipp_.RangeQuery(lo, hi, 0);
    }
    return sum;
  }

  /* insert */
  void Insert(const KeyValue<KeyType>& kv,uint32_t){
    tl_buf_.emplace_back(kv.key,kv.value);
    if(tl_buf_.size()>=hybrid_detail::kStageCap) flush_local();
    uint64_t n=insert_ops_.fetch_add(1,std::memory_order_relaxed)+1;
    if(n%hybrid_detail::kAdaptEvery==0) adapt_cap();
  }

  /* meta */
  bool applicable(bool,bool,bool,bool mt,const std::string&)const{return !mt;}
  std::string name()const{return "HybridPGMLIPPAsync";}
  std::vector<std::string> variants() const { return {SearchClass::name(), std::to_string(pgm_error)}; }
  std::size_t size()const{size_t s=dpgm_.size_in_bytes(); if(lipp_ready_.load()) s+=lipp_.size(); return s;}

 private:
  /* per‑thread staging */
  static thread_local std::vector<KV> tl_buf_;
  void flush_local(){
    std::vector<KV> buf; buf.swap(tl_buf_);
    dp_items_.fetch_add(buf.size(), std::memory_order_relaxed);
    size_t need=0; bool overflow=false;
    {
      std::unique_lock ul(dpgm_m_);
      for(auto &kv:buf) dpgm_.insert(kv.first,kv.second);
      if(dp_items_.load()>buffer_cap_.load()){overflow=true; need=dp_items_.load()-buffer_cap_.load()/2;}
    }
    if(overflow){auto batch=extract_from_dpgm(need); if(!batch.empty()){std::lock_guard lg(q_m_); q_.push(std::move(batch)); cv_.notify_one();}}
  }

  std::vector<KV> extract_from_dpgm(size_t n){
    std::vector<KV> out; std::unique_lock ul(dpgm_m_);
    for(auto it=dpgm_.begin();it!=dpgm_.end()&&out.size()<n;++it) out.emplace_back(it->key(),it->value());
    for(auto& kv:out) dpgm_.erase(kv.first);
    dp_items_.fetch_sub(out.size(), std::memory_order_relaxed);
    return out;
  }

  /* merge */
  void merge_into_lipp(std::vector<KV>& b){
    std::sort(b.begin(),b.end(),[](auto&a,auto&c){return a.first<c.first;});
    b.erase(std::unique(b.begin(),b.end(),[](auto&a,auto&c){return a.first==c.first;}),b.end());
    std::unique_lock ul(lipp_m_);
    if(!lipp_ready_.load()){
      if(b.size()>=hybrid_detail::kMinBulk) {
        std::vector<KeyValue<KeyType>> kv_data;
        kv_data.reserve(b.size());
        for (auto const& kv : b) kv_data.emplace_back(KeyValue<KeyType>{kv.first, kv.second});
        lipp_.Build(kv_data, 0);
        lipp_ready_.store(true);
      }
    }else {
      for (auto& kv : b) {
        KeyValue<KeyType> key_value{kv.first, kv.second};
        lipp_.Insert(key_value, 0);
      }
    }
  }

  /* worker */
  void worker(){
    while(true){
      std::vector<std::vector<KV>> local_batches;
      {
        std::unique_lock ul(q_m_);
        cv_.wait(ul,[this]{return stop_.load()||!q_.empty();});
        if(stop_.load()&&q_.empty()) break;
        while(!q_.empty()){local_batches.emplace_back(std::move(q_.front())); q_.pop();}
      }
      for(auto &b:local_batches) merge_into_lipp(b);
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
  Lipp<KeyType> lipp_{std::vector<int>{}};

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
thread_local std::vector<std::pair<K,uint64_t>> HybridPGMLIPPAsync<K,S,E>::tl_buf_;

namespace tli{
  template<class K,class S,size_t E>
  using HybridPGMLIPPAsync = ::HybridPGMLIPPAsync<K,S,E>;
}

#endif // TLI_HYBRID_PGM_LIPP_ASYNC_H