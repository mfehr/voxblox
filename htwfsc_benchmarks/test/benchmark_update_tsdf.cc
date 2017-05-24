#include <benchmark/benchmark.h>
#include <benchmark_catkin/benchmark_entrypoint.h>

#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"

#include "voxblox_fast/core/tsdf_map.h"
#include "voxblox_fast/integrator/tsdf_integrator.h"

class UpdateTsdfBenchmark : public ::benchmark::Fixture {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  static constexpr double kVoxelSize = 0.01;
  static constexpr size_t kVoxelsPerSide = 16u;
  static constexpr float kTruncationDistance = 0.1;
  static constexpr float kInitialWeight = 0.75;
  static constexpr float kUpdateWeight = 1.75;
  static constexpr float kInitialTsdfValue = 0.0125;
  static constexpr float kUpdateTsdfValue = 0.0175;

  voxblox::TsdfIntegrator::Config config_;
  voxblox_fast::TsdfIntegrator::Config fast_config_;

  voxblox::Point baseline_origin_;
  voxblox_fast::Point fast_origin_;

  voxblox::Point baseline_point_C_;
  voxblox_fast::Point fast_point_C_;

  voxblox::Point baseline_point_G_;
  voxblox_fast::Point fast_point_G_;

  voxblox::Color baseline_color_;
  voxblox_fast::Color fast_color_;

  voxblox::Color baseline_update_color_;
  voxblox_fast::Color fast_update_color_;

  voxblox::Point baseline_voxel_center_;
  voxblox_fast::Point fast_voxel_center_;

  voxblox::BlockIndex baseline_block_idx_;
  voxblox_fast::BlockIndex fast_block_idx_;

  voxblox::TsdfVoxel baseline_voxel_;
  voxblox_fast::TsdfVoxel fast_voxel_;

  std::unique_ptr<voxblox::Layer<voxblox::TsdfVoxel>> baseline_layer_;
  std::unique_ptr<voxblox_fast::Layer<voxblox_fast::TsdfVoxel>> fast_layer_;
  std::unique_ptr<voxblox::TsdfIntegrator> baseline_integrator_;
  std::unique_ptr<voxblox_fast::TsdfIntegrator> fast_integrator_;

  void SetUp(const ::benchmark::State& st) {
    config_.max_ray_length_m = 50.0;
    fast_config_.max_ray_length_m = 50.0;

    baseline_layer_.reset(
        new voxblox::Layer<voxblox::TsdfVoxel>(kVoxelSize, kVoxelsPerSide));
    fast_layer_.reset(new voxblox_fast::Layer<voxblox_fast::TsdfVoxel>(
        kVoxelSize, kVoxelsPerSide));
    baseline_integrator_.reset(
        new voxblox::TsdfIntegrator(config_, baseline_layer_.get()));
    fast_integrator_.reset(
        new voxblox_fast::TsdfIntegrator(fast_config_, fast_layer_.get()));

    fast_block_idx_ = voxblox::BlockIndex(1, 2, 3);
    baseline_block_idx_ = voxblox_fast::BlockIndex(1, 2, 3);

    baseline_origin_ = voxblox::Point(0., 0., 0.);
    fast_origin_ = voxblox_fast::Point(0., 0., 0.);

    baseline_point_C_ = voxblox::Point(1., 2., 3.);
    fast_point_C_ = voxblox_fast::Point(1., 2., 3.);

    baseline_point_G_ = voxblox::Point(2., 3., 4.);
    fast_point_G_ = voxblox_fast::Point(2., 3., 4.);

    baseline_voxel_center_ = voxblox::Point(0.5, 1.5, 2.5);
    fast_voxel_center_ = voxblox_fast::Point(0.5, 1.5, 2.5);

    baseline_color_.r = 100;
    baseline_color_.g = 102;
    baseline_color_.b = 103;

    fast_color_.r = 100;
    fast_color_.g = 102;
    fast_color_.b = 103;

    baseline_voxel_.distance = kInitialTsdfValue;
    baseline_voxel_.weight = kInitialWeight;
    baseline_voxel_.color = baseline_color_;

    fast_voxel_.distance = kInitialTsdfValue;
    fast_voxel_.weight = kInitialWeight;
    fast_voxel_.color = fast_color_;

    baseline_update_color_.r = 201;
    baseline_update_color_.g = 202;
    baseline_update_color_.b = 203;

    fast_update_color_.r = 201;
    fast_update_color_.g = 202;
    fast_update_color_.b = 203;
  }

  void TearDown(const ::benchmark::State&) {
    baseline_layer_.reset();
    fast_layer_.reset();
    baseline_integrator_.reset();
    fast_integrator_.reset();
  }
};

BENCHMARK_DEFINE_F(UpdateTsdfBenchmark, UpdateTsdf_Baseline)
(benchmark::State& state) {
  const size_t num_repetitions = static_cast<double>(state.range(0));
  state.counters["num_updates"] = num_repetitions;
  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      baseline_voxel_.distance = kInitialTsdfValue;
      baseline_voxel_.weight = kInitialWeight;
      baseline_voxel_.color = baseline_color_;
      state.ResumeTiming();

      baseline_integrator_->updateTsdfVoxel(
          baseline_origin_, baseline_point_C_, baseline_point_G_,
          baseline_voxel_center_, baseline_update_color_, kTruncationDistance,
          kUpdateTsdfValue, &baseline_voxel_);
      state.PauseTiming();
    }
  }
}
BENCHMARK_REGISTER_F(UpdateTsdfBenchmark, UpdateTsdf_Baseline)
    ->RangeMultiplier(2)
    ->Range(1, 1e5);

BENCHMARK_DEFINE_F(UpdateTsdfBenchmark, UpdateTsdf_Fast)
(benchmark::State& state) {
  const size_t num_repetitions = static_cast<double>(state.range(0));
  state.counters["num_updates"] = num_repetitions;
  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      fast_voxel_.distance = kInitialTsdfValue;
      fast_voxel_.weight = kInitialWeight;
      fast_voxel_.color = fast_color_;
      state.ResumeTiming();

      fast_integrator_->updateTsdfVoxel(fast_origin_, fast_point_C_,
                                        fast_point_G_, fast_voxel_center_,
                                        fast_update_color_, kTruncationDistance,
                                        kUpdateTsdfValue, &fast_voxel_);
      state.PauseTiming();
    }
  }
}
BENCHMARK_REGISTER_F(UpdateTsdfBenchmark, UpdateTsdf_Fast)
    ->RangeMultiplier(2)
    ->Range(1, 1e5);

BENCHMARKING_ENTRY_POINT
