#include <benchmark/benchmark.h>
#include <benchmark_catkin/benchmark_entrypoint.h>

#include "voxblox/core/common.h"
#include "voxblox_fast/core/common.h"

class BlendColorsBenchmark : public ::benchmark::Fixture {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  void SetUp(const ::benchmark::State& st) {
    baseline_color_1_.r = 100;
    baseline_color_1_.g = 102;
    baseline_color_1_.b = 103;
    baseline_color_1_.a = 104;

    baseline_color_2_.r = 200;
    baseline_color_2_.g = 202;
    baseline_color_2_.b = 203;
    baseline_color_2_.a = 204;

    baseline_color_result_.r = 0;
    baseline_color_result_.g = 0;
    baseline_color_result_.b = 0;
    baseline_color_result_.a = 0;

    fast_color_1_.r = 100;
    fast_color_1_.g = 102;
    fast_color_1_.b = 103;
    fast_color_1_.a = 104;

    fast_color_2_.r = 200;
    fast_color_2_.g = 202;
    fast_color_2_.b = 203;
    fast_color_2_.a = 204;

    fast_color_result_.r = 0;
    fast_color_result_.g = 0;
    fast_color_result_.b = 0;
    fast_color_result_.a = 0;
  }

  void TearDown(const ::benchmark::State&) {}

  voxblox::Color baseline_color_1_;
  float baseline_color_1_weight_;
  voxblox::Color baseline_color_2_;
  float baseline_color_2_weight_;

  voxblox::Color baseline_color_result_;

  voxblox_fast::Color fast_color_1_;
  float fast_color_1_weight_;
  voxblox_fast::Color fast_color_2_;
  float fast_color_2_weight_;

  voxblox_fast::Color fast_color_result_;
};

BENCHMARK_DEFINE_F(BlendColorsBenchmark, ColorBlending_Baseline)
(benchmark::State& state) {
  const size_t num_repetitions = static_cast<double>(state.range(0));
  state.counters["num_blendings"] = num_repetitions;
  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      baseline_color_result_.r = 0;
      baseline_color_result_.g = 0;
      baseline_color_result_.b = 0;
      baseline_color_result_.a = 0;
      state.ResumeTiming();
      baseline_color_result_ = voxblox::Color::blendTwoColors(
          baseline_color_1_, baseline_color_1_weight_, baseline_color_2_,
          baseline_color_2_weight_);
      state.PauseTiming();
    }
  }
}
BENCHMARK_REGISTER_F(BlendColorsBenchmark, ColorBlending_Baseline)
    ->RangeMultiplier(2)
    ->Range(1, 1e5);

BENCHMARK_DEFINE_F(BlendColorsBenchmark, ColorBlending_Fast)
(benchmark::State& state) {
  const size_t num_repetitions = static_cast<double>(state.range(0));
  state.counters["num_blendings"] = num_repetitions;
  while (state.KeepRunning()) {
    state.PauseTiming();
    for (size_t i = 0u; i < num_repetitions; ++i) {
      fast_color_result_.r = 0;
      fast_color_result_.g = 0;
      fast_color_result_.b = 0;
      fast_color_result_.a = 0;
      state.ResumeTiming();
      fast_color_result_ = voxblox_fast::Color::blendTwoColors(
          fast_color_1_, fast_color_1_weight_, fast_color_2_,
          fast_color_2_weight_);
      state.PauseTiming();
    }
  }
}
BENCHMARK_REGISTER_F(BlendColorsBenchmark, ColorBlending_Fast)
    ->RangeMultiplier(2)
    ->Range(1, 1e5);

BENCHMARKING_ENTRY_POINT
