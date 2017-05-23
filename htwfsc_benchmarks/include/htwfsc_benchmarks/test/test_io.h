#ifndef HTWFSC_TEST_IO_H_
#define HTWFSC_TEST_IO_H_

#include <gtest/gtest.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/ply_io.h>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"

#include "voxblox_fast/core/layer.h"
#include "voxblox_fast/core/voxel.h"

namespace htwfsc_benchmarks {
namespace test {

void OutputVoxelsAsPointCloud(
    const voxblox::Layer<voxblox::TsdfVoxel>& layer_A,
    const voxblox_fast::Layer<voxblox_fast::TsdfVoxel>& layer_B,
    const float truncation_distance, const std::string& path_file_A,
    const std::string& path_file_B) {
  const Eigen::Vector3f color_green(0.f, 255.f, 0.f);
  const Eigen::Vector3f color_red(255.f, 0.f, 0.f);
  const Eigen::Vector3f color_blue(0.f, 0.f, 255.f);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_A_ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  voxblox::BlockIndexList blocks_A;
  voxblox_fast::BlockIndexList blocks_B;
  layer_A.getAllAllocatedBlocks(&blocks_A);
  layer_B.getAllAllocatedBlocks(&blocks_B);

  for (const voxblox::BlockIndex& index_A : blocks_A) {
    const voxblox::Block<voxblox::TsdfVoxel>& block_A =
        layer_A.getBlockByIndex(index_A);
    for (size_t voxel_idx = 0u; voxel_idx < block_A.num_voxels(); ++voxel_idx) {
      const voxblox::Point voxel_coordinates =
          block_A.computeCoordinatesFromVoxelIndex(
              block_A.computeVoxelIndexFromLinearIndex(voxel_idx));

      voxblox::TsdfVoxel voxel = block_A.getVoxelByLinearIndex(voxel_idx);

      if (!voxel.weight > 0.f) {
        continue;
      }

      pcl::PointXYZRGB point;
      point.x = voxel_coordinates.x();
      point.y = voxel_coordinates.y();
      point.z = voxel_coordinates.z();

      const float factor = (voxel.distance) / truncation_distance;

      Eigen::Vector3f color;
      if (factor < 0.f) {
        CHECK_GE(factor, -1.f);
        CHECK_LT(factor, 0.f);
        color = color_red - factor * (color_blue - color_red);
      } else {
        CHECK_GE(factor, 0);
        CHECK_LE(factor, 1.f);
        color = color_red + factor * (color_green - color_red);
      }

      uint32_t rgb = (static_cast<uint32_t>(std::round(color.x())) << 16 |
                      static_cast<uint32_t>(std::round(color.y())) << 8 |
                      static_cast<uint32_t>(std::round(color.z())));
      point.rgb = *reinterpret_cast<float*>(&rgb);

      cloud_A_ptr->points.push_back(point);
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_B_ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  for (const voxblox_fast::BlockIndex& index_B : blocks_B) {
    const voxblox_fast::Block<voxblox_fast::TsdfVoxel>& block_B =
        layer_B.getBlockByIndex(index_B);

    for (size_t voxel_idx = 0u; voxel_idx < block_B.num_voxels(); ++voxel_idx) {
      const voxblox_fast::Point voxel_coordinates =
          block_B.computeCoordinatesFromVoxelIndex(
              block_B.computeVoxelIndexFromLinearIndex(voxel_idx));

      const voxblox_fast::TsdfVoxel voxel =
          block_B.getVoxelByLinearIndex(voxel_idx);

      if (!voxel.weight > 0.f) {
        continue;
      }

      pcl::PointXYZRGB point;
      point.x = voxel_coordinates.x();
      point.y = voxel_coordinates.y();
      point.z = voxel_coordinates.z();

      const float factor = (voxel.distance) / truncation_distance;

      Eigen::Vector3f color;
      if (factor < 0.f) {
        CHECK_GE(factor, -1.f);
        CHECK_LT(factor, 0.f);
        color = color_red - factor * (color_blue - color_red);
      } else {
        CHECK_GE(factor, 0);
        CHECK_LE(factor, 1.f);
        color = color_red + factor * (color_green - color_red);
      }

      uint32_t rgb = (static_cast<uint32_t>(std::round(color.x())) << 16 |
                      static_cast<uint32_t>(std::round(color.y())) << 8 |
                      static_cast<uint32_t>(std::round(color.z())));
      point.rgb = *reinterpret_cast<float*>(&rgb);

      cloud_B_ptr->points.push_back(point);
    }
  }

  constexpr bool kBinary = true;
  pcl::PLYWriter ply_writer;
  ply_writer.write(path_file_A, *cloud_A_ptr, kBinary);
  ply_writer.write(path_file_B, *cloud_B_ptr, kBinary);
}
}  // namespace test
}  // namespace htwfsc_benchmarks

#endif  // HTWFSC_TEST_IO_H_
