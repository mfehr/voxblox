#include <benchmark/benchmark.h>
#include <benchmark_catkin/benchmark_entrypoint.h>

#include "voxblox/core/common.h"
#include "voxblox_fast/core/common.h"

#define COUNTFLOPS

class BlendColorsBenchmark : public ::benchmark::Fixture {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  void SetUp(const ::benchmark::State& st) {
    baseline_color_1_.r = 100u;
    baseline_color_1_.g = 102u;
    baseline_color_1_.b = 103u;
    baseline_color_1_.a = 104u;

    baseline_color_2_.r = 200u;
    baseline_color_2_.g = 202u;
    baseline_color_2_.b = 203u;
    baseline_color_2_.a = 204u;

    baseline_color_result_.r = 0u;
    baseline_color_result_.g = 0u;
    baseline_color_result_.b = 0u;
    baseline_color_result_.a = 0u;

    fast_color_1_.rgba[0] = 100u;
    fast_color_1_.rgba[1] = 102u;
    fast_color_1_.rgba[2] = 103u;
    fast_color_1_.rgba[3] = 104u;

    fast_color_2_.rgba[0] = 200u;
    fast_color_2_.rgba[1] = 202u;
    fast_color_2_.rgba[2] = 203u;
    fast_color_2_.rgba[3] = 204u;

    fast_color_result_.rgba[0] = 0u;
    fast_color_result_.rgba[1] = 0u;
    fast_color_result_.rgba[2] = 0u;
    fast_color_result_.rgba[3] = 0u;
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
#ifdef COUNTFLOPS
  state.counters["flops"] =
      1 /*adds*/ + 2 /*divs*/ + 8 /*muls (uint8_t * float)*/ + 4 /*adds*/;
#endif
  while (state.KeepRunning()) {
    baseline_color_result_ = voxblox::Color::blendTwoColors(
        baseline_color_1_, baseline_color_1_weight_, baseline_color_2_,
        baseline_color_2_weight_);
  }
}
BENCHMARK_REGISTER_F(BlendColorsBenchmark, ColorBlending_Baseline);

BENCHMARK_DEFINE_F(BlendColorsBenchmark, ColorBlending_Fast)
(benchmark::State& state) {
#ifdef COUNTFLOPS
  state.counters["flops"] =
      1 /*adds*/ + 2 /*divs*/ + 8 /*muls (uint8_t * float)*/ + 4 /*adds*/;
#endif
  while (state.KeepRunning()) {
    voxblox_fast::Color::blendTwoColors(fast_color_1_, fast_color_1_weight_,
                                        fast_color_2_, fast_color_2_weight_,
                                        &fast_color_result_);
  }
}
BENCHMARK_REGISTER_F(BlendColorsBenchmark, ColorBlending_Fast);

BENCHMARKING_ENTRY_POINT
