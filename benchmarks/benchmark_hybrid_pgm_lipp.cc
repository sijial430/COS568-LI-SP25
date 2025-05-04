#include "benchmarks/benchmark_hybrid_pgm_lipp.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/hybrid_pgm_lipp.h"

template <typename Searcher>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, 
                                 bool pareto, const std::vector<int>& params) {
  std::cout << "Running benchmark with Searcher template" << std::endl;
  if (!pareto) {
    util::fail("Hybrid PGM+LIPP's hyperparameter cannot be set");
  } else {
    std::cout << "Testing with error bounds: 16, 32, 64, 128, 256, 512, 1024" << std::endl;
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 16>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 32>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 64>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 128>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 256>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 512>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 1024>>();
  }
}

template <int record>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, const std::string& filename) {
  std::cout << "Running benchmark with record template for file: " << filename << std::endl;
  if (filename.find("books_100M") != std::string::npos) {
    if (filename.find("0.000000i") != std::string::npos) {
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
    } else if (filename.find("mix") == std::string::npos) {
      if (filename.find("0m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
      } else if (filename.find("1m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
      } else if (filename.find("2m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
      }
    }
  }
  if (filename.find("fb_100M") != std::string::npos) {
    if (filename.find("0.000000i") != std::string::npos) {
      // For initial dataset (no insertions), we test smaller error bounds
      // Binary search is efficient for all segment sizes
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
    } else if (filename.find("mix") == std::string::npos) {
      if (filename.find("0m") != std::string::npos) {
        // For lookups after no insertions, we test larger error bounds
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        
        // Exponential and interpolation search benefit from larger segments
        // where their more complex strategies pay off
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
      } else if (filename.find("1m") != std::string::npos) {
        // After 1M operations, we need larger error bounds for some search strategies
        // as the data structure becomes less optimal
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,1024>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,1024>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,1024>>();
      } else if (filename.find("2m") != std::string::npos) {
        // After 2M operations, we test with even larger error bounds
        // as the data structure has undergone significant changes
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>(); 
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,1024>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,1024>>();  
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,1024>>();
      }
    } else {
      if (filename.find("0.050000i") != std::string::npos) {
        // For low insertion rates (5%), smaller error bounds work well
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();

        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,16>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,64>>();  
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();

        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,16>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        } else if (filename.find("0.500000i") != std::string::npos) {
        // For medium insertion rates (50%), we need larger error bounds
        // as the data structure is more dynamic

        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();

        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
      } else if (filename.find("0.800000i") != std::string::npos || 
                 filename.find("0.900000i") != std::string::npos || 
                 filename.find("0.100000i") != std::string::npos) {
        // For high insertion rates (80-90%), we need the largest error bounds
        // as the data structure is highly dynamic and original model accuracy degrades

        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
      }
    }
  }
  if (filename.find("osmc_100M") != std::string::npos) {
    if (filename.find("0.000000i") != std::string::npos) {
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,16>>();
    } else if (filename.find("mix") == std::string::npos) {
      if (filename.find("0m") != std::string::npos) {
        // benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,1024>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,1024>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,1024>>();
      } else if (filename.find("1m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
      } else if (filename.find("2m") != std::string::npos) {
        // benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,1024>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,1024>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,1024>>();
      }
    } else {
      if (filename.find("0.050000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,16>>();
      } else if (filename.find("0.500000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      } else if (filename.find("0.800000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      }else if (filename.find("0.900000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      } else if (filename.find("0.100000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,256>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      }
    }
  }
  if (filename.find("wiki_100M") != std::string::npos) {
    if (filename.find("0.000000i") != std::string::npos) {
      benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,16>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,32>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
    } else if (filename.find("mix") == std::string::npos) {
      if (filename.find("0m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
      } else if (filename.find("1m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>,512>>();
      } else if (filename.find("2m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,32>>();
      }
    } else {
      if (filename.find("0.050000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,16>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,16>>();
      } else if (filename.find("0.500000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,32>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
      } else if (filename.find("0.800000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      } else if (filename.find("0.900000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      } else if (filename.find("0.100000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>,512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>,128>>();
      }
    }
  }
}

INSTANTIATE_TEMPLATES_MULTITHREAD(benchmark_64_hybrid_pgm_lipp, uint64_t);