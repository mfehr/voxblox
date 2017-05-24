#ifndef VOXBLOX_FAST_CORE_COMMON_H_
#define VOXBLOX_FAST_CORE_COMMON_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <kindr/minimal/quat-transformation.h>
#include <Eigen/Core>

namespace voxblox_fast {
// Types.
typedef float FloatingPoint;
typedef int IndexElement;

typedef Eigen::Matrix<FloatingPoint, 3, 1> Point;
typedef Eigen::Matrix<FloatingPoint, 3, 1> Ray;

typedef Eigen::Matrix<IndexElement, 3, 1> AnyIndex;
typedef AnyIndex VoxelIndex;
typedef AnyIndex BlockIndex;

typedef std::pair<BlockIndex, VoxelIndex> VoxelKey;

typedef std::vector<AnyIndex, Eigen::aligned_allocator<AnyIndex> > IndexVector;
typedef IndexVector BlockIndexList;
typedef IndexVector VoxelIndexList;

struct Color;
typedef uint32_t Label;

// Pointcloud types for external interface.
typedef std::vector<Point, Eigen::aligned_allocator<Point>> Pointcloud;
typedef std::vector<Color> Colors;
typedef std::vector<Label> Labels;

// For triangle meshing/vertex access.
typedef size_t VertexIndex;
typedef std::vector<VertexIndex> VertexIndexList;
typedef Eigen::Matrix<FloatingPoint, 3, 3> Triangle;
typedef std::vector<Triangle, Eigen::aligned_allocator<Triangle> >
    TriangleVector;

// Transformation type for defining sensor orientation.
typedef kindr::minimal::QuatTransformationTemplate<FloatingPoint>
    Transformation;
typedef kindr::minimal::RotationQuaternionTemplate<FloatingPoint> Rotation;

// For alignment of layers / point clouds
typedef Eigen::Matrix<FloatingPoint, 3, Eigen::Dynamic> PointsMatrix;
typedef Eigen::Matrix<FloatingPoint, 3, 3> Matrix3;

// Interpolation structure
typedef Eigen::Matrix<FloatingPoint, 8, 8> InterpTable;
typedef Eigen::Matrix<FloatingPoint, 1, 8> InterpVector;
// Type must allow negatives:
typedef Eigen::Array<IndexElement, 3, 8> InterpIndexes;

// Eigen only provides a SIMD unaryOps from version 3.3 on.
#define EIGEN_HAS_SIMD_UNARY_OPS EIGEN_VERSION_AT_LEAST(2,3,3)

struct Color {
  Color() : r(0), g(0), b(0), a(0) {}
  Color(uint8_t _r, uint8_t _g, uint8_t _b) : Color(_r, _g, _b, 255u) {}
  Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a)
      : r(_r), g(_g), b(_b), a(_a) {}

  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;

  static Color blendTwoColors(const Color& first_color,
                              FloatingPoint first_weight,
                              const Color& second_color,
                              FloatingPoint second_weight) {
    Color new_color;
#if EIGEN_HAS_SIMD_UNARY_OPS
    // Bad hack to avoid refactoring voxblox.
    Eigen::Map<const Eigen::Matrix<uint8_t, 4 , 1>> eigen_color_first(reinterpret_cast<const uint8_t*>(&first_color));
    Eigen::Map<const Eigen::Matrix<uint8_t, 4 , 1>> eigen_color_second(reinterpret_cast<const uint8_t*>(&second_color));

    Eigen::Map<Eigen::Matrix<uint8_t, 4 , 1>> eigen_new_color(reinterpret_cast<uint8_t*>(&new_color));
    eigen_new_color = (first_weight * eigen_color_first.cast<FloatingPoint>() +
    		second_weight * eigen_color_second.cast<FloatingPoint>()).array().round().cast<uint8_t>().matrix();
#else
    FloatingPoint total_weight = first_weight + second_weight;

    first_weight /= total_weight;
    second_weight /= total_weight;

    new_color.r = static_cast<uint8_t>(
        round(first_color.r * first_weight + second_color.r * second_weight));
    new_color.g = static_cast<uint8_t>(
        round(first_color.g * first_weight + second_color.g * second_weight));
    new_color.b = static_cast<uint8_t>(
        round(first_color.b * first_weight + second_color.b * second_weight));
    new_color.a = static_cast<uint8_t>(
        round(first_color.a * first_weight + second_color.a * second_weight));
#endif
    return new_color;
  }

  // Now a bunch of static colors to use! :)
  static const Color White() { return Color(255, 255, 255); }
  static const Color Black() { return Color(0, 0, 0); }
  static const Color Gray() { return Color(127, 127, 127); }
  static const Color Red() { return Color(255, 0, 0); }
  static const Color Green() { return Color(0, 255, 0); }
  static const Color Blue() { return Color(0, 0, 255); }
  static const Color Yellow() { return Color(255, 255, 0); }
  static const Color Orange() { return Color(255, 127, 0); }
  static const Color Purple() { return Color(127, 0, 255); }
  static const Color Teal() { return Color(0, 255, 255); }
  static const Color Pink() { return Color(255, 0, 127); }
};

// Grid <-> point conversion functions.

// IMPORTANT NOTE: Due the limited accuracy of the FloatingPoint type, this
// function doesn't always compute the correct grid index for coordinates
// near the grid cell boundaries.
inline AnyIndex getGridIndexFromPoint(const Point& point,
                                      const FloatingPoint& grid_size_inv) {
#if EIGEN_HAS_SIMD_UNARY_OPS
  return (point.array() * grid_size_inv).floor().cast<IndexElement>();
#else
  return AnyIndex(std::floor(point.x() * grid_size_inv),
		  	  	  std::floor(point.y() * grid_size_inv),
				  std::floor(point.z() * grid_size_inv));
#endif
}

// IMPORTANT NOTE: Due the limited accuracy of the FloatingPoint type, this
// function doesn't always compute the correct grid index for coordinates
// near the grid cell boundaries.
inline AnyIndex getGridIndexFromPoint(const Point& scaled_point) {
  #if EIGEN_HAS_SIMD_UNARY_OPS
    return (scaled_point.array().floor()).cast<IndexElement>();
  #else
	return AnyIndex(std::floor(scaled_point.x()), std::floor(scaled_point.y()),
	                std::floor(scaled_point.z()));
  #endif
}

inline AnyIndex getGridIndexFromOriginPoint(const Point& point,
                                            const FloatingPoint& grid_size_inv) {
#if EIGEN_HAS_SIMD_UNARY_OPS
  return (point.array() * grid_size_inv).round().cast<IndexElement>();
#else
  return AnyIndex(std::round(point.x() * grid_size_inv),
                  std::round(point.y() * grid_size_inv),
                  std::round(point.z() * grid_size_inv));
#endif
}

inline Point getCenterPointFromGridIndex(const AnyIndex& idx,
                                         const FloatingPoint& grid_size) {
  return (Point::Constant(0.5) + idx.cast<FloatingPoint>()) * grid_size;
}

inline Point getOriginPointFromGridIndex(const AnyIndex& idx,
                                         const FloatingPoint& grid_size) {
  return idx.cast<FloatingPoint>() * grid_size;
}

inline BlockIndex getBlockIndexFromGlobalVoxelIndex(
    const AnyIndex& global_voxel_idx, const FloatingPoint& voxels_per_side_inv) {
#if EIGEN_HAS_SIMD_UNARY_OPS
  return (global_voxel_idx.cast<FloatingPoint>().array() * voxels_per_side_inv).floor().cast<IndexElement>();
#else
  return BlockIndex(
    std::floor(static_cast<FloatingPoint>(global_voxel_idx.x()) *
    		   voxels_per_side_inv),
    std::floor(static_cast<FloatingPoint>(global_voxel_idx.y()) *
               voxels_per_side_inv),
    std::floor(static_cast<FloatingPoint>(global_voxel_idx.z()) *
               voxels_per_side_inv));
#endif
}

inline VoxelIndex getLocalFromGlobalVoxelIndex(const AnyIndex& global_voxel_idx,
                                               int voxels_per_side) {
  VoxelIndex local_voxel_idx = global_voxel_idx.unaryExpr(
		  [voxels_per_side](const IndexElement global_index) {
	IndexElement local_index = global_index % voxels_per_side;
	if (local_index < 0) {
		local_index += voxels_per_side;
	}
	return local_index;
  });
  return local_voxel_idx;
}

// Math functions.
inline int signum(const FloatingPoint& x) { return (x == 0) ? 0 : x < 0 ? -1 : 1; }

// For occupancy/octomap-style mapping.
inline float logOddsFromProbability(float probability) {
  DCHECK(probability >= 0.0f && probability <= 1.0f);
  return log(probability / (1.0 - probability));
}

inline float probabilityFromLogOdds(float log_odds) {
  return 1.0 - (1.0 / (1.0 + exp(log_odds)));
}

template <typename Type, typename... Arguments>
inline std::shared_ptr<Type> aligned_shared(Arguments&&... arguments) {
  typedef typename std::remove_const<Type>::type TypeNonConst;
  return std::allocate_shared<Type>(Eigen::aligned_allocator<TypeNonConst>(),
                                    std::forward<Arguments>(arguments)...);
}
}  // namespace voxblox

#endif  // VOXBLOX_FAST_CORE_COMMON_H_
