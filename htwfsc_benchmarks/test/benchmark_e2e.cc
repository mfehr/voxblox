#include <memory>

#include <benchmark/benchmark.h>
#include <benchmark_catkin/benchmark_entrypoint.h>

#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"

#include "voxblox_fast/core/tsdf_map.h"
#include "voxblox_fast/integrator/tsdf_integrator.h"

#include "htwfsc_benchmarks/simulation/sphere_simulator.h"

#define COUNTFLOPS

#ifdef COUNTFLOPS
extern flopcounter countflops;
#endif

class E2EBenchmark : public ::benchmark::Fixture {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  void SetUp(const ::benchmark::State& /*state*/) {
    config_.max_ray_length_m = 50.0;
    fast_config_.max_ray_length_m = 50.0;
    config_.use_weight_dropoff = false;
    fast_config_.use_weight_dropoff = false;

    baseline_layer_.reset(
        new voxblox::Layer<voxblox::TsdfVoxel>(kVoxelSize, kVoxelsPerSide));
    fast_layer_.reset(new voxblox_fast::Layer<voxblox_fast::TsdfVoxel>(
        kVoxelSize, kVoxelsPerSide));
    baseline_integrator_.reset(
        new voxblox::TsdfIntegrator(config_, baseline_layer_.get()));
    fast_integrator_.reset(
        new voxblox_fast::TsdfIntegrator(fast_config_, fast_layer_.get()));
    T_G_C = voxblox::Transformation();

    kNumBlocksPerSide = std::abs(kMinIdx - kMaxIdx);
    kNumBlocks =
      kNumBlocksPerSide * kNumBlocksPerSide * kNumBlocksPerSide;
  }

  void CreateSphere(const double radius, const size_t num_points) {
    sphere_points_C.clear();
    htwfsc_benchmarks::sphere_sim::createSphere(kMean, kSigma, radius,
                                                num_points, &sphere_points_C);
    colors_.clear();
    colors_.resize(sphere_points_C.size(), voxblox::Color(128, 253, 5));
    fast_colors_.clear();
    fast_colors_.resize(sphere_points_C.size(),
                        voxblox_fast::Color(128, 253, 5));
  }

  void TearDown(const ::benchmark::State& /*state*/) {
    baseline_layer_.reset();
    fast_layer_.reset();
    baseline_integrator_.reset();
    fast_integrator_.reset();

    sphere_points_C.clear();
    colors_.clear();
  }

  template <typename LayerType>
  void AllocateBlocksInIndexRange(const int min_idx, const int max_idx,
                                  LayerType* layer) {
    CHECK_NOTNULL(layer);
    for (int x = min_idx; x < max_idx; ++x) {
      for (int y = min_idx; y < max_idx; ++y) {
        for (int z = min_idx; z < max_idx; ++z) {
          Eigen::Matrix<int, 3, 1> index = Eigen::Matrix<int, 3, 1>(x, y, z);
          layer->allocateBlockPtrByIndex(index);
        }
      }
    }
  }

  size_t GetMemoryUsage(const int num_blocks) {
    voxblox::Block<voxblox::TsdfVoxel> block(kVoxelsPerSide, kVoxelSize,
                                             voxblox::Point());
    const size_t memory_bytes = num_blocks * block.getMemorySize();
    return memory_bytes;
  }

  voxblox::Colors colors_;
  voxblox_fast::Colors fast_colors_;
  voxblox::Pointcloud sphere_points_C;
  voxblox::Transformation T_G_C;

  static constexpr double kVoxelSize = 0.01;
  static constexpr size_t kVoxelsPerSide = 16u;
  static constexpr double kMean = 0;
  static constexpr double kSigma = 0.05;

  // Params for the benchmarks with varying radius.
  static constexpr size_t kNumPoints = 1000u;

  // Params for the benchmarks with varying number of points.
  static constexpr double kRadius = 1.0;
  static constexpr int kMinIdx = -(kRadius / (kVoxelsPerSide * kVoxelSize) + 1);
  static constexpr int kMaxIdx = (kRadius / (kVoxelsPerSide * kVoxelSize) + 1);
  int kNumBlocksPerSide;
  int kNumBlocks;

  voxblox::TsdfIntegrator::Config config_;
  voxblox_fast::TsdfIntegrator::Config fast_config_;

  std::unique_ptr<voxblox::TsdfIntegrator> baseline_integrator_;
  std::unique_ptr<voxblox_fast::TsdfIntegrator> fast_integrator_;

  std::unique_ptr<voxblox::Layer<voxblox::TsdfVoxel>> baseline_layer_;
  std::unique_ptr<voxblox_fast::Layer<voxblox_fast::TsdfVoxel>> fast_layer_;
};

//////////////////////////////////////////////////////////////
// BENCHMARK CONSTANT NUMBER OF POINTS WITH CHANGING RADIUS //
//////////////////////////////////////////////////////////////

BENCHMARK_DEFINE_F(E2EBenchmark, Radius_Baseline)(benchmark::State& state) {
  // Create test data.
  const double radius_cm = static_cast<double>(state.range(0)) / 2.;
  state.counters["radius_cm"] = radius_cm;
  const double radius_m = radius_cm / 100;
  CreateSphere(radius_m, kNumPoints);

  // Compute the amount of memory/blocks needed for the computation.
  const int min_idx = -(radius_m / (kVoxelsPerSide * kVoxelSize) + 1);
  const int max_idx = (radius_m / (kVoxelsPerSide * kVoxelSize) + 1);
  const int num_blocks_per_side = std::abs(max_idx - min_idx);
  const int num_blocks =
      num_blocks_per_side * num_blocks_per_side * num_blocks_per_side;
  state.counters["preallocated_memory_B"] = GetMemoryUsage(num_blocks);
  state.counters["preallocated_blocks"] = num_blocks;

  // NOTE(mfehr): Either preallocate here and reuse the same blocks everytime we
  // iterate below, or completely deallocate and allocate in for every
  // iteration.
  AllocateBlocksInIndexRange(min_idx, max_idx, baseline_layer_.get());

#ifdef COUNTFLOPS
  countflops.ResetCastRay();
  countflops.ResetUpdateTsdf();
  baseline_integrator_->integratePointCloud_flopcount(T_G_C, sphere_points_C,
                                                      colors_);
  size_t flops = countflops.castray_adds + countflops.castray_divs;
  flops += countflops.updatetsdf_adds + countflops.updatetsdf_muls +
           countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
  state.counters["flops"] = flops;
#endif

  while (state.KeepRunning()) {
    // state.PauseTiming();
    // baseline_layer_->removeAllBlocks();
    // AllocateBlocksInIndexRange(min_idx, max_idx, baseline_layer_.get());
    // // Make sure all memory operations are finished.
    // benchmark::ClobberMemory();
    // state.ResumeTiming();

    baseline_integrator_->integratePointCloud(T_G_C, sphere_points_C, colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, Radius_Baseline)->DenseRange(100, 900, 50);

BENCHMARK_DEFINE_F(E2EBenchmark, Radius_Fast)(benchmark::State& state) {
  // Create test data.
  const double radius_cm = static_cast<double>(state.range(0)) / 2.;
  state.counters["radius_cm"] = radius_cm;
  const double radius_m = radius_cm / 100;
  CreateSphere(radius_m, kNumPoints);

  // Compute the amount of memory/blocks needed for the computation.
  const int min_idx = -(radius_m / (kVoxelsPerSide * kVoxelSize) + 1);
  const int max_idx = (radius_m / (kVoxelsPerSide * kVoxelSize) + 1);
  const int num_blocks_per_side = std::abs(max_idx - min_idx);
  const int num_blocks =
      num_blocks_per_side * num_blocks_per_side * num_blocks_per_side;
  state.counters["preallocated_memory_B"] = GetMemoryUsage(num_blocks);
  state.counters["preallocated_blocks"] = num_blocks;

  // NOTE(mfehr): Either preallocate here and reuse the same blocks everytime we
  // iterate below, or completely deallocate and allocate in for every
  // iteration.
  AllocateBlocksInIndexRange(min_idx, max_idx, fast_layer_.get());

  #ifdef COUNTFLOPS
    countflops.ResetCastRay();
    countflops.ResetUpdateTsdf();
    baseline_integrator_->integratePointCloud_flopcount(T_G_C, sphere_points_C,
                                                        colors_);
    size_t flops = countflops.castray_adds + countflops.castray_divs;
    flops += countflops.updatetsdf_adds + countflops.updatetsdf_muls +
             countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
    state.counters["flops"] = flops;
  #endif

  while (state.KeepRunning()) {
    // state.PauseTiming();
    // fast_layer_->removeAllBlocks();
    // AllocateBlocksInIndexRange(min_idx, max_idx, fast_layer_.get());
    // // Make sure all memory operations are finished.
    // benchmark::ClobberMemory();
    // state.ResumeTiming();

    fast_integrator_->integratePointCloud(T_G_C, sphere_points_C, fast_colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, Radius_Fast)->DenseRange(100, 900, 50);

//////////////////////////////////////////////////////////////
// BENCHMARK CONSTANT RADIUS WITH CHANGING NUMBER OF POINTS //
//////////////////////////////////////////////////////////////

BENCHMARK_DEFINE_F(E2EBenchmark, NumPoints_Baseline)
(benchmark::State& state) {
  const size_t num_points = static_cast<double>(state.range(0));
  CreateSphere(kRadius, num_points);
  state.counters["num_points"] = sphere_points_C.size();

  // Print the amount of memory/blocks needed for the computation.
  state.counters["preallocated_memory_B"] = GetMemoryUsage(kNumBlocks);
  state.counters["preallocated_blocks"] = kNumBlocks;

  // NOTE(mfehr): Either preallocate here and reuse the same blocks everytime we
  // iterate below, or completely deallocate and allocate in for every
  // iteration.
  AllocateBlocksInIndexRange(kMinIdx, kMaxIdx, baseline_layer_.get());

  #ifdef COUNTFLOPS
    countflops.ResetCastRay();
    countflops.ResetUpdateTsdf();
    baseline_integrator_->integratePointCloud_flopcount(T_G_C, sphere_points_C,
                                                        colors_);
    size_t flops = countflops.castray_adds + countflops.castray_divs;
    flops += countflops.updatetsdf_adds + countflops.updatetsdf_muls +
             countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
    state.counters["flops"] = flops;
  #endif

  while (state.KeepRunning()) {
    // state.PauseTiming();
    // baseline_layer_->removeAllBlocks();
    // AllocateBlocksInIndexRange(kMinIdx, kMaxIdx, baseline_layer_.get());
    // // Make sure all memory operations are finished.
    // benchmark::ClobberMemory();
    // state.ResumeTiming();

    baseline_integrator_->integratePointCloud(T_G_C, sphere_points_C, colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, NumPoints_Baseline)
    ->RangeMultiplier(2)
    ->Range(16, 1e7);

BENCHMARK_DEFINE_F(E2EBenchmark, NumPoints_Fast)(benchmark::State& state) {
  const size_t num_points = static_cast<double>(state.range(0));
  CreateSphere(kRadius, num_points);
  state.counters["num_points"] = sphere_points_C.size();

  // Print the amount of memory/blocks needed for the computation.
  state.counters["preallocated_memory_B"] = GetMemoryUsage(kNumBlocks);
  state.counters["preallocated_blocks"] = kNumBlocks;

  // NOTE(mfehr): Either preallocate here and reuse the same blocks everytime we
  // iterate below, or completely deallocate and allocate in for every
  // iteration.
  AllocateBlocksInIndexRange(kMinIdx, kMaxIdx, fast_layer_.get());

  #ifdef COUNTFLOPS
    countflops.ResetCastRay();
    countflops.ResetUpdateTsdf();
    baseline_integrator_->integratePointCloud_flopcount(T_G_C, sphere_points_C,
                                                        colors_);
    size_t flops = countflops.castray_adds + countflops.castray_divs;
    flops += countflops.updatetsdf_adds + countflops.updatetsdf_muls +
             countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
    state.counters["flops"] = flops;
  #endif

  while (state.KeepRunning()) {
    // state.PauseTiming();
    // fast_layer_->removeAllBlocks();
    // AllocateBlocksInIndexRange(kMinIdx, kMaxIdx, fast_layer_.get());
    // // Make sure all memory operations are finished.
    // benchmark::ClobberMemory();
    // state.ResumeTiming();

    fast_integrator_->integratePointCloud(T_G_C, sphere_points_C, fast_colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, NumPoints_Fast)
    ->RangeMultiplier(2)
    ->Range(16, 1e7);

BENCHMARKING_ENTRY_POINT
