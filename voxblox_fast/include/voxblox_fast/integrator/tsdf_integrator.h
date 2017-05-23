#ifndef VOXBLOX_FAST_INTEGRATOR_TSDF_INTEGRATOR_H_
#define VOXBLOX_FAST_INTEGRATOR_TSDF_INTEGRATOR_H_

#include <algorithm>
#include <vector>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>

#include <Eigen/Core>
#include <glog/logging.h>

#include "voxblox_fast/core/layer.h"
#include "voxblox_fast/core/voxel.h"
#include "voxblox_fast/integrator/integrator_utils.h"
#include "voxblox_fast/utils/timing.h"

namespace voxblox_fast {

class TsdfIntegrator {
 public:
  struct Config {
    float default_truncation_distance = 0.1;
    float max_weight = 10000.0;
    bool voxel_carving_enabled = true;
    FloatingPoint min_ray_length_m = 0.1;
    FloatingPoint max_ray_length_m = 5.0;
    bool use_const_weight = false;
    bool allow_clear = true;
    bool use_weight_dropoff = true;
    size_t integrator_threads = std::thread::hardware_concurrency();
  };

  struct VoxelInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TsdfVoxel voxel;

    BlockIndex block_idx;
    VoxelIndex local_voxel_idx;

    Point point_C;
    Point point_G;
  };

  TsdfIntegrator(const Config& config, Layer<TsdfVoxel>* layer)
      : config_(config), layer_(layer) {
    DCHECK(layer_);

    voxel_size_ = layer_->voxel_size();
    block_size_ = layer_->block_size();
    voxels_per_side_ = layer_->voxels_per_side();

    voxel_size_inv_ = 1.0 / voxel_size_;
    block_size_inv_ = 1.0 / block_size_;
    voxels_per_side_inv_ = 1.0 / voxels_per_side_;

    if (config_.integrator_threads == 0) {
      LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
      config_.integrator_threads = 1;
    }
  }

  float getVoxelWeight(const Point& point_C, const Point& point_G,
                       const Point& origin, const Point& voxel_center) const {
    if (config_.use_const_weight) {
      return 1.0;
    }
    FloatingPoint dist_z = std::abs(point_C.z());
    if (dist_z > 1e-6) {
      return 1.0 / (dist_z * dist_z);
    }
    return 0.0;
  }

  inline __m128 dotProducts(const __m128 v, const __m128 v0, const __m128 v1,
                            const __m128 v2, const __m128 v3) {
    __m128 vv0 = _mm_mul_ps(v, v0);
    __m128 vv1 = _mm_mul_ps(v, v1);
    __m128 vv2 = _mm_mul_ps(v, v2);
    __m128 vv3 = _mm_mul_ps(v, v3);

    __m128 vv0_vv1 = _mm_hadd_ps(vv0, vv1);
    __m128 vv2_vv3 = _mm_hadd_ps(vv2, vv3);

    return _mm_hadd_ps(vv0_vv1, vv2_vv3);
  }

  inline void updateTsdfVoxelSse(
      const Point& voxel0_center, const Point& voxel1_center,
      const Point& voxel2_center, const Point& voxel3_center,
      const __m128 vec_origin, const __m128 vec_v_point_origin,
      const __m128 vec_dist_G, const __m128 vec_trunc_dist_pos,
      const __m128 vec_trunc_dist_neg, const __m128 vec_weight,
      const float weight, const Color& color, TsdfVoxel* tsdf_voxel0,
      TsdfVoxel* tsdf_voxel1, TsdfVoxel* tsdf_voxel2, TsdfVoxel* tsdf_voxel3) {
    __m128 v_voxel0_center = loadPointToSse(voxel0_center);
    __m128 v_voxel1_center = loadPointToSse(voxel1_center);
    __m128 v_voxel2_center = loadPointToSse(voxel2_center);
    __m128 v_voxel3_center = loadPointToSse(voxel3_center);

    __m128 v_voxel0_origin = _mm_sub_ps(v_voxel0_center, vec_origin);
    __m128 v_voxel1_origin = _mm_sub_ps(v_voxel1_center, vec_origin);
    __m128 v_voxel2_origin = _mm_sub_ps(v_voxel2_center, vec_origin);
    __m128 v_voxel3_origin = _mm_sub_ps(v_voxel3_center, vec_origin);

    __m128 dots = dotProducts(vec_v_point_origin, v_voxel0_origin,
                              v_voxel1_origin, v_voxel2_origin,
                              v_voxel3_origin);
    __m128 dist_G_V = _mm_div_ps(dots, vec_dist_G);

    __m128 voxel_weights = _mm_set_ps(tsdf_voxel3->weight, tsdf_voxel2->weight,
                                      tsdf_voxel1->weight, tsdf_voxel0->weight);
    __m128 voxel_distances = _mm_set_ps(tsdf_voxel3->distance,
                                        tsdf_voxel2->distance,
                                        tsdf_voxel1->distance,
                                        tsdf_voxel0->distance);
    __m128 new_weight = _mm_add_ps(vec_weight, voxel_weights);

    __m128 sdf = _mm_sub_ps(vec_dist_G, dist_G_V);
    __m128 upd_sdf = _mm_mul_ps(sdf, vec_weight);
    __m128 cur_sdf = _mm_mul_ps(voxel_distances, voxel_weights);
    __m128 new_sdf_temp = _mm_add_ps(upd_sdf, cur_sdf);
    __m128 new_sdf = _mm_div_ps(new_sdf_temp, new_weight);

    __m128 new_sdf_lim_up = _mm_min_ps(new_sdf, vec_trunc_dist_pos);
    __m128 new_sdf_lim = _mm_max_ps(new_sdf_lim_up, vec_trunc_dist_neg);

    float sdf_array[4];
    _mm_store_ps(sdf_array, new_sdf_lim);

    __m128 weight_limit = _mm_set1_ps(config_.max_weight);
    __m128 weight_capped = _mm_min_ps(new_weight, weight_limit);
    float weight_array[4];
    _mm_store_ps(weight_array, weight_capped);

    /*Point voxel0_origin = voxel0_center - origin;
    FloatingPoint sdist_G_V = voxel0_origin.dot(v_point_origin) / dist_G;
    float sdf_s = static_cast<float>(dist_G - sdist_G_V);
    const float new_weight_s = tsdf_voxel0->weight + weight;
    const float new_sdf_s =
        (sdf_s * new_weight_s + tsdf_voxel0->distance * tsdf_voxel0->weight) /
        new_weight_s;
    const float new_sdf_lim_s = (new_sdf_s > 0.0)
                                   ? std::min(truncation_distance, new_sdf_s)
                                   : std::max(-truncation_distance, new_sdf_s);*/

    __m128 voxel_weights_scaled = _mm_div_ps(voxel_weights, new_weight);
    __m128 new_weights_scaled = _mm_div_ps(vec_weight, new_weight);

    float voxel_weights_scaled_array[4];
    _mm_store_ps(voxel_weights_scaled_array, voxel_weights_scaled);
    float new_weights_scaled_array[4];
    _mm_store_ps(new_weights_scaled_array, new_weights_scaled);

    tsdf_voxel0->color = Color::blendTwoColorsWithScaledWeights(
        tsdf_voxel0->color, voxel_weights_scaled_array[0], color,
        new_weights_scaled_array[0]);
    tsdf_voxel0->distance = sdf_array[0];
    tsdf_voxel0->weight = weight_array[0];

    tsdf_voxel1->color = Color::blendTwoColorsWithScaledWeights(
        tsdf_voxel1->color, voxel_weights_scaled_array[1], color,
        new_weights_scaled_array[1]);
    tsdf_voxel1->distance = sdf_array[1];
    tsdf_voxel1->weight = weight_array[1];

    tsdf_voxel2->color = Color::blendTwoColorsWithScaledWeights(
        tsdf_voxel2->color, voxel_weights_scaled_array[2], color,
        new_weights_scaled_array[2]);
    tsdf_voxel2->distance = sdf_array[2];
    tsdf_voxel2->weight = weight_array[2];

    tsdf_voxel3->color = Color::blendTwoColorsWithScaledWeights(
        tsdf_voxel3->color, voxel_weights_scaled_array[3], color,
        new_weights_scaled_array[3]);
    tsdf_voxel3->distance = sdf_array[3];
    tsdf_voxel3->weight = weight_array[3];
  }

  inline void updateTsdfVoxel(const Point& origin,
                              const Point& v_point_origin, const FloatingPoint& dist_G,
                              const Point& voxel_center,
                              const Color& color,
                              const float truncation_distance,
                              const float weight, TsdfVoxel* tsdf_voxel) {
    // Figure out whether the voxel is behind or in front of the surface.
    // To do this, project the voxel_center onto the ray from origin to point G.
    // Then check if the the magnitude of the vector is smaller or greater than
    // the original distance...
    Point v_voxel_origin = voxel_center - origin;

    // projection of a (v_voxel_origin) onto b (v_point_origin)
    FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

    float sdf = static_cast<float>(dist_G - dist_G_V);

    float updated_weight = weight;
    // Compute updated weight in case we use weight dropoff. It's easier here
    // that in getVoxelWeight as here we have the actual SDF for the voxel
    // already computed.
    CHECK(!config_.use_weight_dropoff);
    /*const FloatingPoint dropoff_epsilon = voxel_size_;
    if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
      updated_weight = weight * (truncation_distance + sdf) /
                       (truncation_distance - dropoff_epsilon);
      updated_weight = std::max(updated_weight, 0.0f);
    }*/

    const float new_weight = tsdf_voxel->weight + updated_weight;
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
    const float new_sdf =
        (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
        new_weight;

    tsdf_voxel->distance = (new_sdf > 0.0)
                               ? std::min(truncation_distance, new_sdf)
                               : std::max(-truncation_distance, new_sdf);
    tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
  }

  inline float computeDistance(const Point& origin, const Point& point_G,
                               const Point& voxel_center) {
    Point v_voxel_origin = voxel_center - origin;
    Point v_point_origin = point_G - origin;

    FloatingPoint dist_G = v_point_origin.norm();
    // projection of a (v_voxel_origin) onto b (v_point_origin)
    FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

    float sdf = static_cast<float>(dist_G - dist_G_V);
    return sdf;
  }

  void integratePointCloud(const Transformation& T_G_C,
                           const Pointcloud& points_C, const Colors& colors) {
    DCHECK_EQ(points_C.size(), colors.size());

    //timing::Timing::Reset();

    timing::Timer integrate_timer("integrate");

    const Point& origin = T_G_C.getPosition();

    for (size_t pt_idx = 0; pt_idx < points_C.size(); ++pt_idx) {
      const Point& point_C = points_C[pt_idx];
      const Point point_G = T_G_C * point_C;
      const Color& color = colors[pt_idx];

      FloatingPoint ray_distance = (point_G - origin).norm();
      if (ray_distance < config_.min_ray_length_m) {
        continue;
      } else if (ray_distance > config_.max_ray_length_m) {
        // TODO(helenol): clear until max ray length instead.
        continue;
      }

      FloatingPoint truncation_distance = config_.default_truncation_distance;

      const Ray unit_ray = (point_G - origin).normalized();

      const Point ray_end = point_G + unit_ray * truncation_distance;
      const Point ray_start = config_.voxel_carving_enabled
                                  ? origin
                                  : (point_G - unit_ray * truncation_distance);

      const Point start_scaled = ray_start * voxel_size_inv_;
      const Point end_scaled = ray_end * voxel_size_inv_;

      IndexVector global_voxel_indices;
      timing::Timer cast_ray_timer("integrate/cast_ray");
      castRay(start_scaled, end_scaled, &global_voxel_indices);
      cast_ray_timer.Stop();

      timing::Timer update_voxels_timer("integrate/update_voxels");

      BlockIndex last_block_idx = BlockIndex::Zero();
      Block<TsdfVoxel>::Ptr block0;
      Block<TsdfVoxel>::Ptr block1;
      Block<TsdfVoxel>::Ptr block2;
      Block<TsdfVoxel>::Ptr block3;

      const float weight =
          getVoxelWeight(point_C, point_G, origin, /*dummy=*/origin);

      const Point v_point_origin = point_G - origin;
      const FloatingPoint dist_G = v_point_origin.norm();
      const __m128 vec_dist_G = _mm_set1_ps(dist_G);

      const __m128 vec_origin = loadPointToSse(origin);
      const __m128 vec_v_point_origin = loadPointToSse(v_point_origin);

      const __m128 vec_trunc_dist_pos = _mm_set1_ps(truncation_distance);
      const __m128 vec_trunc_dist_neg = _mm_set1_ps(-truncation_distance);

      const __m128 vec_weight = _mm_set1_ps(weight);

      const int limit = global_voxel_indices.size() - 3;
      int i;
      for (i = 0; i < limit; i += 4) {
        //timing::Timer i1_timer("integrate/i1");
        BlockIndex block0_idx = getBlockIndexFromGlobalVoxelIndex(
            global_voxel_indices[i], voxels_per_side_inv_);
        BlockIndex block1_idx = getBlockIndexFromGlobalVoxelIndex(
            global_voxel_indices[i + 1], voxels_per_side_inv_);
        BlockIndex block2_idx = getBlockIndexFromGlobalVoxelIndex(
            global_voxel_indices[i + 2], voxels_per_side_inv_);
        BlockIndex block3_idx = getBlockIndexFromGlobalVoxelIndex(
            global_voxel_indices[i + 3], voxels_per_side_inv_);
        //i1_timer.Stop();

        //timing::Timer i2_timer("integrate/i2");
        VoxelIndex local_voxel0_idx =
            getLocalFromGlobalVoxelIndex(global_voxel_indices[i],
                                         voxels_per_side_);
        VoxelIndex local_voxel1_idx =
            getLocalFromGlobalVoxelIndex(global_voxel_indices[i + 1],
                                         voxels_per_side_);
        VoxelIndex local_voxel2_idx =
            getLocalFromGlobalVoxelIndex(global_voxel_indices[i + 2],
                                         voxels_per_side_);
        VoxelIndex local_voxel3_idx =
            getLocalFromGlobalVoxelIndex(global_voxel_indices[i + 3],
                                         voxels_per_side_);
        //i2_timer.Stop();

        if (!block0 || block0_idx != last_block_idx) {
          block0 = layer_->allocateBlockPtrByIndex(block0_idx);
          block0->updated() = true;
        } else {
          block0 = block3;
          block0->updated() = true;
        }

        if (block1_idx == last_block_idx && block3) {
          block1 = block3;
          block1->updated() = true;
        } else if (block1_idx == block0_idx) {
          block1 = block0;
          block1->updated() = true;
        } else {
          block1 = layer_->allocateBlockPtrByIndex(block1_idx);
          block1->updated() = true;
        }

        if (block2_idx == last_block_idx && block3) {
          block2 = block3;
          block2->updated() = true;
        } else if (block2_idx == block0_idx) {
          block2 = block0;
          block2->updated() = true;
        } else if (block2_idx == block1_idx) {
          block2 = block1;
          block2->updated() = true;
        } else {
          block2 = layer_->allocateBlockPtrByIndex(block2_idx);
          block2->updated() = true;
        }

        if (block3_idx == last_block_idx && block3) {
          block3->updated() = true;
        } else if (block3_idx == block0_idx) {
          block3 = block0;
          block3->updated() = true;
        } else if (block3_idx == block1_idx) {
          block3 = block1;
          block3->updated() = true;
        } else if (block3_idx == block2_idx) {
          block3 = block2;
          block3->updated() = true;
        } else {
          block3 = layer_->allocateBlockPtrByIndex(block3_idx);
          block3->updated() = true;
        }
        last_block_idx = block3_idx;

        //timing::Timer i3_timer("integrate/i3");
        const Point voxel0_center_G =
            block0->computeCoordinatesFromVoxelIndex(local_voxel0_idx);
        const Point voxel1_center_G =
            block1->computeCoordinatesFromVoxelIndex(local_voxel1_idx);
        const Point voxel2_center_G =
            block2->computeCoordinatesFromVoxelIndex(local_voxel2_idx);
        const Point voxel3_center_G =
            block3->computeCoordinatesFromVoxelIndex(local_voxel3_idx);
        //i3_timer.Stop();

        //timing::Timer i4_timer("integrate/i4");
        TsdfVoxel& tsdf_voxel0 = block0->getVoxelByVoxelIndex(local_voxel0_idx);
        TsdfVoxel& tsdf_voxel1 = block1->getVoxelByVoxelIndex(local_voxel1_idx);
        TsdfVoxel& tsdf_voxel2 = block2->getVoxelByVoxelIndex(local_voxel2_idx);
        TsdfVoxel& tsdf_voxel3 = block3->getVoxelByVoxelIndex(local_voxel3_idx);
        //i4_timer.Stop();

        updateTsdfVoxelSse(voxel0_center_G, voxel1_center_G, voxel2_center_G,
                           voxel3_center_G, vec_origin, vec_v_point_origin, vec_dist_G,
                           vec_trunc_dist_pos, vec_trunc_dist_neg, vec_weight, weight,
                           color, &tsdf_voxel0,
                           &tsdf_voxel1, &tsdf_voxel2, &tsdf_voxel3);
      }

      for (; i < global_voxel_indices.size(); ++i) {
        BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(
            global_voxel_indices[i], voxels_per_side_inv_);
        VoxelIndex local_voxel_idx =
            getLocalFromGlobalVoxelIndex(global_voxel_indices[i],
                                         voxels_per_side_);

        if (!block3 || block_idx != last_block_idx) {
          block3 = layer_->allocateBlockPtrByIndex(block_idx);
          block3->updated() = true;
          last_block_idx = block_idx;
        }

        const Point voxel_center_G =
            block3->computeCoordinatesFromVoxelIndex(local_voxel_idx);
        TsdfVoxel& tsdf_voxel = block3->getVoxelByVoxelIndex(local_voxel_idx);

        updateTsdfVoxel(origin, v_point_origin, dist_G, voxel_center_G, color,
                        truncation_distance, weight, &tsdf_voxel);
      }

      update_voxels_timer.Stop();
    }
    integrate_timer.Stop();

    //std::cout << timing::Timing::Print();
  }

  inline void bundleRays(
      const Transformation& T_G_C, const Pointcloud& points_C,
      BlockHashMapType<std::vector<size_t>>::type* voxel_map,
      BlockHashMapType<std::vector<size_t>>::type* clear_map) {
    for (size_t pt_idx = 0; pt_idx < points_C.size(); ++pt_idx) {
      const Point& point_C = points_C[pt_idx];
      const Point point_G = T_G_C * point_C;

      FloatingPoint ray_distance = (point_C).norm();
      if (ray_distance < config_.min_ray_length_m) {
        continue;
      } else if (config_.allow_clear &&
                 ray_distance > config_.max_ray_length_m) {
        VoxelIndex voxel_index =
            getGridIndexFromPoint(point_G, voxel_size_inv_);
        (*clear_map)[voxel_index].push_back(pt_idx);
        continue;
      }

      // Figure out what the end voxel is here.
      VoxelIndex voxel_index = getGridIndexFromPoint(point_G, voxel_size_inv_);
      (*voxel_map)[voxel_index].push_back(pt_idx);
    }

    LOG(INFO) << "Went from " << points_C.size() << " points to "
              << voxel_map->size() << " raycasts  and " << clear_map->size()
              << " clear rays.";
  }

  void updateVoxel(const VoxelInfo& voxel_info, const Point& origin) {
    static BlockIndex last_block_idx = BlockIndex::Zero();
    static Block<TsdfVoxel>::Ptr block;

    if (!block || voxel_info.block_idx != last_block_idx) {
      block = layer_->allocateBlockPtrByIndex(voxel_info.block_idx);
      block->updated() = true;
      last_block_idx = voxel_info.block_idx;
    }

    const Point voxel_center_G =
        block->computeCoordinatesFromVoxelIndex(voxel_info.local_voxel_idx);
    TsdfVoxel& tsdf_voxel =
        block->getVoxelByVoxelIndex(voxel_info.local_voxel_idx);

    LOG(FATAL) << "Not implemented.";
   /* updateTsdfVoxel(origin, voxel_info.point_C, voxel_info.point_G,
                    voxel_center_G, voxel_info.voxel.color,
                    config_.default_truncation_distance,
                    voxel_info.voxel.weight, &tsdf_voxel);*/
  }

  void integrateVoxel(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, bool discard, bool clearing_ray,
      const std::pair<AnyIndex, std::vector<size_t>>& kv,
      const BlockHashMapType<std::vector<size_t>>::type& voxel_map,
      std::queue<VoxelInfo>* voxel_update_queue) {
    if (kv.second.empty()) {
      return;
    }

    const Point& origin = T_G_C.getPosition();
    const Point voxel_center_offset(0.5, 0.5, 0.5);

    // stores all the information needed to update a map voxel
    VoxelInfo voxel_info;
    voxel_info.point_C = Point::Zero();
    voxel_info.voxel.weight = 0.0;

    for (const size_t pt_idx : kv.second) {
      const Point& point_C = points_C[pt_idx];
      const Color& color = colors[pt_idx];

      float point_weight = getVoxelWeight(
          point_C, T_G_C * point_C, origin,
          (kv.first.cast<FloatingPoint>() + voxel_center_offset) * voxel_size_);
      voxel_info.point_C = (voxel_info.point_C * voxel_info.voxel.weight +
                            point_C * point_weight) /
                           (voxel_info.voxel.weight + point_weight);
      voxel_info.voxel.color = Color::blendTwoColors(
          voxel_info.voxel.color, voxel_info.voxel.weight, color, point_weight);
      voxel_info.voxel.weight += point_weight;

      // only take first point when clearing
      if (clearing_ray) {
        break;
      }
    }

    voxel_info.point_G = T_G_C * voxel_info.point_C;
    const Ray unit_ray = (voxel_info.point_G - origin).normalized();

    Point ray_end, ray_start;
    if (clearing_ray) {
      ray_end = origin + unit_ray * config_.max_ray_length_m;
      ray_start = origin;
    } else {
      ray_end =
          voxel_info.point_G + unit_ray * config_.default_truncation_distance;
      ray_start = config_.voxel_carving_enabled
                      ? origin
                      : (voxel_info.point_G -
                         unit_ray * config_.default_truncation_distance);
    }

    const Point start_scaled = ray_start * voxel_size_inv_;
    const Point end_scaled = ray_end * voxel_size_inv_;

    IndexVector global_voxel_index;
    timing::Timer cast_ray_timer("integrate/cast_ray");
    castRay(start_scaled, end_scaled, &global_voxel_index);
    cast_ray_timer.Stop();

    timing::Timer update_voxels_timer("integrate/update_voxels");

    for (const AnyIndex& global_voxel_idx : global_voxel_index) {
      if (discard) {
        // Check if this one is already the the block hash map for this
        // insertion. Skip this to avoid grazing.
        if ((clearing_ray || global_voxel_idx != kv.first) &&
            voxel_map.find(global_voxel_idx) != voxel_map.end()) {
          continue;
        }
      }

      voxel_info.block_idx = getGridIndexFromPoint(
          global_voxel_idx.cast<FloatingPoint>(), voxels_per_side_inv_);

      voxel_info.local_voxel_idx =
          getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_);

      voxel_update_queue->push(voxel_info);
    }
    update_voxels_timer.Stop();
  }

  void integrateVoxels(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, bool discard, bool clearing_ray,
      const BlockHashMapType<std::vector<size_t>>::type& voxel_map,
      const BlockHashMapType<std::vector<size_t>>::type& clear_map,
      std::queue<VoxelInfo>* voxel_update_queue, size_t tid) {
    BlockHashMapType<std::vector<size_t>>::type::const_iterator it;
    size_t map_size;
    if (clearing_ray) {
      it = clear_map.begin();
      map_size = clear_map.size();
    } else {
      it = voxel_map.begin();
      map_size = voxel_map.size();
    }

    for (size_t i = 0; i < map_size; ++i) {
      if (((i + tid + 1) % config_.integrator_threads) == 0) {
        integrateVoxel(T_G_C, points_C, colors, discard, clearing_ray, *it,
                       voxel_map, voxel_update_queue);
      }
      ++it;
    }
  }

  void integrateRays(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, bool discard, bool clearing_ray,
      const BlockHashMapType<std::vector<size_t>>::type& voxel_map,
      const BlockHashMapType<std::vector<size_t>>::type& clear_map) {
    std::vector<std::queue<VoxelInfo>> voxel_update_queues(
        config_.integrator_threads);

    const Point& origin = T_G_C.getPosition();

    // if only 1 thread just do function call, otherwise spawn threads
    if (config_.integrator_threads == 1) {
      integrateVoxels(T_G_C, points_C, colors, discard, clearing_ray, voxel_map,
                      clear_map, &(voxel_update_queues[0]), 0);
    } else {
      std::vector<std::thread> integration_threads;
      for (size_t i = 0; i < config_.integrator_threads; ++i) {
        integration_threads.emplace_back(&TsdfIntegrator::integrateVoxels, this,
                                         T_G_C, points_C, colors, discard,
                                         clearing_ray, voxel_map, clear_map,
                                         &(voxel_update_queues[i]), i);
      }

      for (std::thread& thread : integration_threads) {
        thread.join();
      }
    }
    for (std::queue<VoxelInfo>& voxel_update_queue : voxel_update_queues) {
      while (!voxel_update_queue.empty()) {
        updateVoxel(voxel_update_queue.front(), origin);
        voxel_update_queue.pop();
      }
    }
  }

  void integratePointCloudMerged(const Transformation& T_G_C,
                                 const Pointcloud& points_C,
                                 const Colors& colors, bool discard) {
    DCHECK_EQ(points_C.size(), colors.size());
    timing::Timer integrate_timer("integrate");

    // Pre-compute a list of unique voxels to end on.
    // Create a hashmap: VOXEL INDEX -> index in original cloud.
    BlockHashMapType<std::vector<size_t>>::type voxel_map;
    // This is a hash map (same as above) to all the indices that need to be
    // cleared.
    BlockHashMapType<std::vector<size_t>>::type clear_map;

    bundleRays(T_G_C, points_C, &voxel_map, &clear_map);

    integrateRays(T_G_C, points_C, colors, discard, false, voxel_map,
                  clear_map);

    timing::Timer clear_timer("integrate/clear");

    integrateRays(T_G_C, points_C, colors, discard, true, voxel_map, clear_map);

    clear_timer.Stop();

    integrate_timer.Stop();
  }

  // Returns a CONST ref of the config.
  const Config& getConfig() const { return config_; }

 protected:
  Config config_;

  Layer<TsdfVoxel>* layer_;

  // Cached map config.
  FloatingPoint voxel_size_;
  size_t voxels_per_side_;
  FloatingPoint block_size_;

  // Derived types.
  FloatingPoint voxel_size_inv_;
  FloatingPoint voxels_per_side_inv_;
  FloatingPoint block_size_inv_;
};

}  // namespace voxblox

#endif  // VOXBLOX_FAST_INTEGRATOR_TSDF_INTEGRATOR_H_
