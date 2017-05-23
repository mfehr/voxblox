#ifndef VOXBLOX_FAST_INTEGRATOR_INTEGRATOR_UTILS_H_
#define VOXBLOX_FAST_INTEGRATOR_INTEGRATOR_UTILS_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Core>

#include "voxblox_fast/core/common.h"
#include "voxblox_fast/utils/timing.h"

namespace voxblox_fast {
inline bool findIntersectionPoint(const Point& start, const Ray& ray,
                                  const Point& min, const Point& max,
                                  Point* intersection_point) {
  DCHECK(intersection_point != nullptr);

  const Point invdir = 1.0 / ray.array();
  const AnyIndex sign = invdir.unaryExpr(std::ptr_fun(isneg));
  Point bounds[2];
  bounds[0] = min;
  bounds[1] = max;

  // CHECK(!isnan(invdir.x()))
  //     << "invdir: " << invdir.transpose() << " ray: " << ray.transpose();
  // CHECK(!isnan(invdir.y()))
  //     << "invdir: " << invdir.transpose() << " ray: " << ray.transpose();
  // CHECK(!isnan(invdir.z()))
  //     << "invdir: " << invdir.transpose() << " ray: " << ray.transpose();
  //
  // CHECK(!isnan(sign.x())) << "sign: " << sign.transpose();
  // CHECK(!isnan(sign.y())) << "sign: " << sign.transpose();
  // CHECK(!isnan(sign.z())) << "sign: " << sign.transpose();

  float tmin, tmax, tymin, tymax, tzmin, tzmax, t;

  // The min and max are there to filter out the cases where the bound - start
  // // is zero and the inverse is +-inf, i.e.:
  // 0 * +/-inf  = nan.
  // tmin = std::max(0., (bounds[sign.x()].x() - start.x()) * invdir.x());
  // tmax = std::min(1., (bounds[1 - sign.x()].x() - start.x()) * invdir.x());
  // tymin = std::max(0., (bounds[sign.y()].y() - start.y()) * invdir.y());
  // tymax = std::min(1., (bounds[1 - sign.y()].y() - start.y()) * invdir.y());

  tmin = (bounds[sign.x()].x() - start.x()) * invdir.x();
  tmax = (bounds[1 - sign.x()].x() - start.x()) * invdir.x();
  tymin = (bounds[sign.y()].y() - start.y()) * invdir.y();
  tymax = (bounds[1 - sign.y()].y() - start.y()) * invdir.y();

  // CHECK(!isnan(tmin)) << "tmin: " << tmin << " sign: " << sign.transpose()
  //                     << " invdir: " << invdir.transpose();
  // CHECK(!isnan(tmax)) << "tmax: " << tmax;
  // CHECK(!isnan(tymin)) << "tymin: " << tymin;
  // CHECK(!isnan(tymax)) << "tymax: " << tymax;

  if ((tmin > tymax) || (tymin > tmax)) {
    return false;
  }

  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;

  // tzmin = std::max(0., (bounds[sign.z()].z() - start.z()) * invdir.z());
  // tzmax = std::min(1., (bounds[1 - sign.z()].z() - start.z()) * invdir.z());
  tzmin = (bounds[sign.z()].z() - start.z()) * invdir.z();
  tzmax = (bounds[1 - sign.z()].z() - start.z()) * invdir.z();

  if ((tmin > tzmax) || (tzmin > tmax)) {
    return false;
  }

  if (tzmin > tmin) tmin = tzmin;
  if (tzmax < tmax) tmax = tzmax;

  t = tmin;

  if (t < 0) {
    t = tmax;
    if (t < 0) {
      return false;
    }
  }

  *intersection_point = start.array() + tmin * ray.array();
  return true;
}

// This function returns a list of local voxel indices within a volume of voxels
// defined by min_index and max_index.
inline void castRayInVolume(
    const Point& start_scaled, const Point& end_scaled,
    const AnyIndex& min_index, const AnyIndex& max_index,
    std::vector<AnyIndex, Eigen::aligned_allocator<AnyIndex> >* indices) {
  CHECK_NOTNULL(indices);

  constexpr FloatingPoint kTolerance = 1e-6;

  AnyIndex curr_index = getGridIndexFromPoint(start_scaled);
  const AnyIndex end_index = getGridIndexFromPoint(end_scaled);
  Ray ray_scaled = end_scaled - start_scaled;

  // Once we enter the volume for the first time we set this to true. If we are
  // inside the volume and encounter a voxel that is outside again, we know we
  // can stop ray casting.
  bool entered_volume = false;
  if (curr_index.x() >= min_index.x() && curr_index.x() <= max_index.x() &&
      curr_index.y() >= min_index.y() && curr_index.y() <= max_index.y() &&
      curr_index.z() >= min_index.z() && curr_index.z() <= max_index.z()) {
    indices->push_back(curr_index - min_index);
    entered_volume = true;
  } else {
    // If we are not already inside the volume, compute intersection of ray with
    // volume to get a new start index.
    timing::Timer find_intersection(
        "integrate/block_ray_casting/voxel_ray_casting/intersection");
    Point start_intersection;
    CHECK(findIntersectionPoint(
        start_scaled, ray_scaled, min_index.cast<FloatingPoint>(),
        max_index.cast<FloatingPoint>().array() + 1.0, &start_intersection));
    find_intersection.Stop();

    curr_index = getGridIndexFromPoint(start_intersection);

    ray_scaled = end_scaled - start_intersection;

    // Check if we are already inside the volume. Due to rounding errors, this
    // is not necessarily true for the intersection point. If we are not already
    // inside the volume, we must reach it with the next step.
    if (curr_index.x() >= min_index.x() && curr_index.x() <= max_index.x() &&
        curr_index.y() >= min_index.y() && curr_index.y() <= max_index.y() &&
        curr_index.z() >= min_index.z() && curr_index.z() <= max_index.z()) {
      indices->push_back(curr_index - min_index);
      entered_volume = true;
    }
  }

  // If we are already done, abort early.
  if (curr_index == end_index) {
    return;
  }

  timing::Timer ray_casting(
      "integrate/block_ray_casting/voxel_ray_casting/ray_casting");
  // Prepare variables for ray casting.
  const AnyIndex ray_step_signs = ray_scaled.unaryExpr(std::ptr_fun(signum));

  const AnyIndex corrected_step(std::max(0, ray_step_signs.x()),
                                std::max(0, ray_step_signs.y()),
                                std::max(0, ray_step_signs.z()));

  const Point start_scaled_shifted =
      start_scaled - curr_index.cast<FloatingPoint>();

  const Ray distance_to_boundaries =
      corrected_step.cast<FloatingPoint>() - start_scaled_shifted;

  Ray t_to_next_boundary((std::abs(ray_scaled.x()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.x() / ray_scaled.x(),
                         (std::abs(ray_scaled.y()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.y() / ray_scaled.y(),
                         (std::abs(ray_scaled.z()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.z() / ray_scaled.z());

  // Distance to cross one grid cell along the ray in t.
  // Same as absolute inverse value of delta_coord.
  const Ray t_step_size =
      ray_step_signs.cast<FloatingPoint>().cwiseQuotient(ray_scaled);

  while (curr_index != end_index) {
    int t_min_idx;
    t_to_next_boundary.minCoeff(&t_min_idx);
    DCHECK_LT(t_min_idx, 3);
    DCHECK_GE(t_min_idx, 0);

    curr_index[t_min_idx] += ray_step_signs[t_min_idx];
    t_to_next_boundary[t_min_idx] += t_step_size[t_min_idx];

    if (curr_index.x() >= min_index.x() && curr_index.x() <= max_index.x() &&
        curr_index.y() >= min_index.y() && curr_index.y() <= max_index.y() &&
        curr_index.z() >= min_index.z() && curr_index.z() <= max_index.z()) {
      indices->push_back(curr_index - min_index);
      entered_volume = true;
    } else {
      if (entered_volume) {
        ray_casting.Stop();
        return;
      } else {
        // CHECK(false) << "There is something wrong with the ray tracing! There
        // "
        //           << "can be no more than one additional ray tracing step "
        //           << "before entering the volume, otherwise the "
        //           << "intersection point calculation is flawed.";
      }
    }
  }
  ray_casting.Stop();
}

// This function assumes PRE-SCALED coordinates, where one unit = one voxel
// size. The indices are also returned in this scales coordinate system, which
// should map to Local/Voxel indices.
inline void castRay(
    const Point& start_scaled, const Point& end_scaled,
    std::vector<AnyIndex, Eigen::aligned_allocator<AnyIndex> >* indices) {
  CHECK_NOTNULL(indices);

  constexpr FloatingPoint kTolerance = 1e-6;

  const AnyIndex start_index = getGridIndexFromPoint(start_scaled);
  const AnyIndex end_index = getGridIndexFromPoint(end_scaled);
  indices->push_back(start_index);

  if (start_index == end_index) {
    return;
  }

  const Ray ray_scaled = end_scaled - start_scaled;

  const AnyIndex ray_step_signs = ray_scaled.unaryExpr(std::ptr_fun(signum));

  const AnyIndex corrected_step(std::max(0, ray_step_signs.x()),
                                std::max(0, ray_step_signs.y()),
                                std::max(0, ray_step_signs.z()));

  const Point start_scaled_shifted =
      start_scaled - start_index.cast<FloatingPoint>();

  const Ray distance_to_boundaries =
      corrected_step.cast<FloatingPoint>() - start_scaled_shifted;

  Ray t_to_next_boundary((std::abs(ray_scaled.x()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.x() / ray_scaled.x(),
                         (std::abs(ray_scaled.y()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.y() / ray_scaled.y(),
                         (std::abs(ray_scaled.z()) < kTolerance)
                             ? 2.0
                             : distance_to_boundaries.z() / ray_scaled.z());

  // Distance to cross one grid cell along the ray in t.
  // Same as absolute inverse value of delta_coord.
  const Ray t_step_size =
      ray_step_signs.cast<FloatingPoint>().cwiseQuotient(ray_scaled);

  AnyIndex curr_index = start_index;
  while (curr_index != end_index) {
    int t_min_idx;
    t_to_next_boundary.minCoeff(&t_min_idx);
    DCHECK_LT(t_min_idx, 3);
    DCHECK_GE(t_min_idx, 0);

    curr_index[t_min_idx] += ray_step_signs[t_min_idx];
    t_to_next_boundary[t_min_idx] += t_step_size[t_min_idx];

    indices->push_back(curr_index);
  }
}

// Takes start and end in WORLD COORDINATES, does all pre-scaling and
// sorting into hierarhical index.
inline void getHierarchicalIndexAlongRay(
    const Point& start, const Point& end, size_t voxels_per_side,
    FloatingPoint voxel_size, FloatingPoint truncation_distance,
    bool voxel_carving_enabled, HierarchicalIndexMap* hierarchical_idx_map) {
  hierarchical_idx_map->clear();

  FloatingPoint voxels_per_side_inv = 1.0 / voxels_per_side;
  FloatingPoint voxel_size_inv = 1.0 / voxel_size;

  const Ray unit_ray = (end - start).normalized();

  const Point ray_end = end + unit_ray * truncation_distance;
  const Point ray_start =
      voxel_carving_enabled ? start : (end - unit_ray * truncation_distance);

  const Point start_scaled = ray_start * voxel_size_inv;
  const Point end_scaled = ray_end * voxel_size_inv;

  IndexVector global_voxel_index;
  timing::Timer cast_ray_timer("integrate/cast_ray");
  castRay(start_scaled, end_scaled, &global_voxel_index);
  cast_ray_timer.Stop();

  timing::Timer create_index_timer("integrate/create_hi_index");
  for (const AnyIndex& global_voxel_idx : global_voxel_index) {
    BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(
        global_voxel_idx, voxels_per_side_inv);
    VoxelIndex local_voxel_idx =
        getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side);

    if (local_voxel_idx.x() < 0) {
      local_voxel_idx.x() += voxels_per_side;
    }
    if (local_voxel_idx.y() < 0) {
      local_voxel_idx.y() += voxels_per_side;
    }
    if (local_voxel_idx.z() < 0) {
      local_voxel_idx.z() += voxels_per_side;
    }

    (*hierarchical_idx_map)[block_idx].push_back(local_voxel_idx);
  }
  create_index_timer.Stop();
}

}  // namespace voxblox

#endif  // VOXBLOX_FAST_INTEGRATOR_INTEGRATOR_UTILS_H_
