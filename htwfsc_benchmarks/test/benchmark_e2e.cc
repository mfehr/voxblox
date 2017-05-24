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
  void AllocateSetOfBlocks(const int min_idx, const int max_idx,
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

  voxblox::Colors colors_;
  voxblox_fast::Colors fast_colors_;
  voxblox::Pointcloud sphere_points_C;
  voxblox::Transformation T_G_C;

  static constexpr double kVoxelSize = 0.01;
  static constexpr size_t kVoxelsPerSide = 16u;

  static constexpr double kMean = 0;
  static constexpr double kSigma = 0.05;
  static constexpr size_t kNumPoints = 100000u;
  static constexpr double kRadius = 2.0;

  static constexpr int kMinIdx = -(kRadius / (kVoxelsPerSide * kVoxelSize) + 2);
  static constexpr int kMaxIdx = (kRadius / (kVoxelsPerSide * kVoxelSize) + 2);

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
  const double radius = static_cast<double>(state.range(0)) / 2.0;
  state.counters["radius_cm"] = radius * 100;
  CreateSphere(radius, kNumPoints);
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
    state.PauseTiming();
    baseline_layer_->removeAllBlocks();
    AllocateSetOfBlocks(kMinIdx, kMaxIdx, baseline_layer_.get());
    state.ResumeTiming();
    baseline_integrator_->integratePointCloud(T_G_C, sphere_points_C, colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, Radius_Baseline)->DenseRange(1, 3, 1);

BENCHMARK_DEFINE_F(E2EBenchmark, Radius_Fast)(benchmark::State& state) {
  const double radius = static_cast<double>(state.range(0)) / 2.0;
  state.counters["radius_cm"] = radius * 100;
  CreateSphere(radius, kNumPoints);
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
    state.PauseTiming();
    fast_layer_->removeAllBlocks();
    AllocateSetOfBlocks(kMinIdx, kMaxIdx, fast_layer_.get());
    state.ResumeTiming();
    fast_integrator_->integratePointCloud(T_G_C, sphere_points_C, fast_colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, Radius_Fast)->DenseRange(1, 3, 1);

//////////////////////////////////////////////////////////////
// BENCHMARK CONSTANT RADIUS WITH CHANGING NUMBER OF POINTS //
//////////////////////////////////////////////////////////////

BENCHMARK_DEFINE_F(E2EBenchmark, NumPoints_Baseline)
(benchmark::State& state) {
  const size_t num_points = static_cast<double>(state.range(0));
  CreateSphere(kRadius, num_points);
  state.counters["num_points"] = sphere_points_C.size();
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
    state.PauseTiming();
    baseline_layer_->removeAllBlocks();
    AllocateSetOfBlocks(kMinIdx, kMaxIdx, baseline_layer_.get());
    state.ResumeTiming();
    baseline_integrator_->integratePointCloud(T_G_C, sphere_points_C, colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, NumPoints_Baseline)
    ->RangeMultiplier(2)
    ->Range(1, 1e4);

BENCHMARK_DEFINE_F(E2EBenchmark, NumPoints_Fast)(benchmark::State& state) {
  const size_t num_points = static_cast<double>(state.range(0));
  CreateSphere(kRadius, num_points);
  state.counters["num_points"] = sphere_points_C.size();
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
    state.PauseTiming();
    fast_layer_->removeAllBlocks();
    AllocateSetOfBlocks(kMinIdx, kMaxIdx, fast_layer_.get());
    state.ResumeTiming();
    fast_integrator_->integratePointCloud(T_G_C, sphere_points_C, fast_colors_);
  }
}
BENCHMARK_REGISTER_F(E2EBenchmark, NumPoints_Fast)
    ->RangeMultiplier(2)
    ->Range(1, 1e4);

BENCHMARKING_ENTRY_POINT
