#include <benchmark/benchmark.h>
#include <benchmark_catkin/benchmark_entrypoint.h>

#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"

#include "voxblox_fast/core/tsdf_map.h"
#include "voxblox_fast/integrator/tsdf_integrator.h"

#define COUNTFLOPS

#ifdef COUNTFLOPS
extern flopcounter countflops;
#endif

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
  voxblox_fast::TsdfVoxel fast_voxel_1_;
  voxblox_fast::TsdfVoxel fast_voxel_2_;
  voxblox_fast::TsdfVoxel fast_voxel_3_;
  voxblox_fast::TsdfVoxel fast_voxel_4_;

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
    baseline_color_.a = 104;

    fast_color_.rgba[0] = 101;
    fast_color_.rgba[1] = 102;
    fast_color_.rgba[2] = 103;
    fast_color_.rgba[3] = 104;

    baseline_voxel_.distance = kInitialTsdfValue;
    baseline_voxel_.weight = kInitialWeight;
    baseline_voxel_.color = baseline_color_;

    fast_voxel_1_.distance = kInitialTsdfValue;
    fast_voxel_1_.weight = kInitialWeight;
    fast_voxel_1_.color = fast_color_;

    fast_voxel_2_ = fast_voxel_1_;
    fast_voxel_3_ = fast_voxel_1_;
    fast_voxel_4_ = fast_voxel_1_;

    baseline_update_color_.r = 201;
    baseline_update_color_.g = 202;
    baseline_update_color_.b = 203;
    baseline_update_color_.a = 204;

    fast_update_color_.rgba[0] = 201;
    fast_update_color_.rgba[1] = 202;
    fast_update_color_.rgba[2] = 203;
    fast_update_color_.rgba[3] = 204;
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

#ifdef COUNTFLOPS
  baseline_voxel_.distance = kInitialTsdfValue;
  baseline_voxel_.weight = kInitialWeight;
  baseline_voxel_.color = baseline_color_;

  countflops.ResetUpdateTsdf();
  baseline_integrator_->updateTsdfVoxel_flopcount(
      baseline_origin_, baseline_point_C_, baseline_point_G_,
      baseline_voxel_center_, baseline_update_color_, kTruncationDistance,
      kUpdateTsdfValue, &baseline_voxel_);

  size_t flops = countflops.updatetsdf_adds + countflops.updatetsdf_muls +
                 countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
  state.counters["flops"] = flops * 4 * num_repetitions;
#endif

  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      baseline_voxel_.distance = kInitialTsdfValue;
      baseline_voxel_.weight = kInitialWeight;
      baseline_voxel_.color = baseline_color_;
      // Make sure all memory operations are finished.
      benchmark::ClobberMemory();
      state.ResumeTiming();

      // We need to run this 4 times, because the fast version has been
      // vectorized to update  4 voxels at once.
      for (size_t j = 0; j < 4; ++j) {
        baseline_integrator_->updateTsdfVoxel(
            baseline_origin_, baseline_point_C_, baseline_point_G_,
            baseline_voxel_center_, baseline_update_color_, kTruncationDistance,
            kUpdateTsdfValue, &baseline_voxel_);
      }
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

#ifdef COUNTFLOPS
  baseline_voxel_.distance = kInitialTsdfValue;
  baseline_voxel_.weight = kInitialWeight;
  baseline_voxel_.color = baseline_color_;

  countflops.ResetUpdateTsdf();
  baseline_integrator_->updateTsdfVoxel_flopcount(
      baseline_origin_, baseline_point_C_, baseline_point_G_,
      baseline_voxel_center_, baseline_update_color_, kTruncationDistance,
      kUpdateTsdfValue, &baseline_voxel_);

  size_t flops = countflops.updatetsdf_adds + countflops.updatetsdf_muls +
                 countflops.updatetsdf_divs + countflops.updatetsdf_sqrts;
  state.counters["flops"] = flops * 4 * num_repetitions;
#endif

  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      // Reset data.
      fast_voxel_1_.distance = kInitialTsdfValue;
      fast_voxel_1_.weight = kInitialWeight;
      fast_voxel_1_.color = fast_color_;
      fast_voxel_2_ = fast_voxel_1_;
      fast_voxel_3_ = fast_voxel_1_;
      fast_voxel_4_ = fast_voxel_1_;
      // Make sure all memory operations are finished.
      benchmark::ClobberMemory();
      state.ResumeTiming();

      // Prepare vectorized data:
      const __m128 vec_weight = _mm_set1_ps(kUpdateWeight);

      const __m128 vec_trunc_dist_pos = _mm_set1_ps(kTruncationDistance);
      const __m128 vec_trunc_dist_neg = _mm_set1_ps(-kTruncationDistance);

      const __m128 vec_origin = voxblox_fast::loadPointToSse(fast_origin_);

      const voxblox_fast::Point v_point_origin = fast_point_G_ - fast_origin_;
      const voxblox_fast::FloatingPoint dist_G = v_point_origin.norm();
      const __m128 vec_dist_G = _mm_set1_ps(dist_G);

      const __m128 vec_v_point_origin =
          voxblox_fast::loadPointToSse(v_point_origin);

      fast_integrator_->updateTsdfVoxelSse(
          fast_voxel_center_, fast_voxel_center_, fast_voxel_center_,
          fast_voxel_center_, vec_origin, vec_v_point_origin, vec_dist_G,
          vec_trunc_dist_pos, vec_trunc_dist_neg, vec_weight, fast_color_,
          &fast_voxel_1_, &fast_voxel_2_, &fast_voxel_3_, &fast_voxel_4_);

      state.PauseTiming();
    }
  }
}
BENCHMARK_REGISTER_F(UpdateTsdfBenchmark, UpdateTsdf_Fast)
    ->RangeMultiplier(2)
    ->Range(1, 1e5);

BENCHMARKING_ENTRY_POINT
