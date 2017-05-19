#include <random>

#include <eigen-checks/entrypoint.h>
#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/integrator_utils.h"

#include "voxblox_fast/core/tsdf_map.h"
#include "voxblox_fast/integrator/integrator_utils.h"

#include "htwfsc_benchmarks/simulation/sphere_simulator.h"

static constexpr size_t kSeed = 242u;

class FastCastRayTest : public ::testing::Test {
 public:
  // Test data params.
  static constexpr double kMean = 0;
  static constexpr double kSigma = 0.05;
  static constexpr int kNumPoints = 1000;
  static constexpr double kRadius = 2.0;
  static constexpr size_t kNumDifferentSpheres = 10u;

 protected:
  virtual void SetUp() {
    std::default_random_engine gen(kSeed);
    std::normal_distribution<double> translation_norm_dist(0.0, 2.0);
    std::normal_distribution<double> angle_dist(0.0,
                                                2.0 * 3.141592653589793238463);
    T_G_C_vector_.clear();
    sphere_points_G_vector_.clear();

    T_G_C_vector_.resize(kNumDifferentSpheres);
    sphere_points_G_vector_.resize(kNumDifferentSpheres);

    for (size_t sphere_idx = 0u; sphere_idx < kNumDifferentSpheres;
         ++sphere_idx) {
      htwfsc_benchmarks::sphere_sim::createSphere(
          kMean, kSigma, kRadius, kNumPoints,
          &(sphere_points_G_vector_[sphere_idx]));

      T_G_C_vector_[sphere_idx].setRandom(translation_norm_dist(gen),
                                          angle_dist(gen));

      // Transform to global frame.
      const voxblox::Transformation& T_G_C = T_G_C_vector_[sphere_idx];
      for (voxblox::Point& point : sphere_points_G_vector_[sphere_idx]) {
        point = T_G_C.transform(point);
      }
    }
  }

  std::vector<voxblox::Pointcloud> sphere_points_G_vector_;
  std::vector<voxblox::Transformation> T_G_C_vector_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

TEST_F(FastCastRayTest, CompareToBaseline) {
  std::vector<voxblox::AnyIndex, Eigen::aligned_allocator<voxblox::AnyIndex>> indices_baseline;
  std::vector<voxblox::AnyIndex, Eigen::aligned_allocator<voxblox::AnyIndex>> indices_fast;
  for (size_t sphere_idx = 0u; sphere_idx < kNumDifferentSpheres;
       ++sphere_idx) {
    const voxblox::Point& origin = T_G_C_vector_[sphere_idx].getPosition();

    for (const voxblox::Point& sphere_point : sphere_points_G_vector_[sphere_idx]) {
      indices_baseline.clear();
      voxblox::castRay(origin, sphere_point, &indices_baseline);

      indices_fast.clear();
      voxblox_fast::castRay(origin, sphere_point, &indices_fast);
      ASSERT_EQ(indices_baseline, indices_fast);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  int result = RUN_ALL_TESTS();

  return result;
}
