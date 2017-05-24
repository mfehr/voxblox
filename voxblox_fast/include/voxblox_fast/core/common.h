#ifndef VOXBLOX_FAST_CORE_COMMON_H_
#define VOXBLOX_FAST_CORE_COMMON_H_

#include <immintrin.h>
#include <smmintrin.h>

#include <iostream>
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

typedef std::vector<AnyIndex, Eigen::aligned_allocator<AnyIndex>> IndexVector;
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
typedef std::vector<Triangle, Eigen::aligned_allocator<Triangle>>
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

#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__((aligned(x)))
#endif
#endif

inline __m128 loadPointToSse(const Point& point) {
  __m128 p_x = _mm_load_ss(&point(0));
  __m128 p_y = _mm_load_ss(&point(1));
  __m128 p_z = _mm_load_ss(&point(2));
  // Combine x and y first.
  __m128 p_xy = _mm_movelh_ps(p_x, p_y);
  // Reshuffle all 3 values into the correct places.
  return _mm_shuffle_ps(p_xy, p_z, _MM_SHUFFLE(2, 0, 2, 0));
}

// This could potentially be faster, but it seems there's no guarantee
// regarding the alignment of Vector3f of eigen. It causes weird memory
// issues.
/*inline __m128 loadPointToSse(const Point& point) {
  __m128i xy = _mm_loadl_epi64((const __m128i*)point.data());
  __m128 z = _mm_load_ss(&point(2));
  return _mm_movelh_ps(_mm_castsi128_ps(xy), z);
}*/

typedef union {
  __m128 v;
  int a[4];
} ui;
inline void printVec4i(__m128 v, char const* name) {
  ui u;
  u.v = v;
  printf("Vector %s: [ %d\t%d\t%d\t%d ]\n", name, u.a[0], u.a[1], u.a[2],
         u.a[3]);
}

typedef union {
  __m128 v;
  float a[4];
} uf;
inline void printVec4(__m128 v, char const* name) {
  uf u;
  u.v = v;
  printf("Vector %s: [ %f\t%f\t%f\t%f ]\n", name, u.a[0], u.a[1], u.a[2],
         u.a[3]);
}

struct Color {
  Color() {
    rgba[0] = 0;
    rgba[1] = 0;
    rgba[2] = 0;
    rgba[3] = 0;
  }
  Color(uint8_t _r, uint8_t _g, uint8_t _b) : Color(_r, _g, _b, 255) {}
  Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) {
    rgba[0] = _r;
    rgba[1] = _g;
    rgba[2] = _b;
    rgba[3] = _a;
  }

  uint8_t rgba[4];

  static void blendTwoColors(const Color& first_color,
                             FloatingPoint first_weight,
                             const Color& second_color,
                             FloatingPoint second_weight, Color* new_color) {
    const FloatingPoint total_weight = first_weight + second_weight;

    first_weight /= total_weight;
    second_weight /= total_weight;

    __m128i c1v = _mm_setr_epi32(first_color.rgba[0], first_color.rgba[1],
                                 first_color.rgba[2], first_color.rgba[3]);
    __m128i c2v = _mm_setr_epi32(second_color.rgba[0], second_color.rgba[1],
                                 second_color.rgba[2], second_color.rgba[3]);

    __m128 c1fv = _mm_cvtepi32_ps(c1v);
    __m128 c2fv = _mm_cvtepi32_ps(c2v);

    __m128 weight_1 = _mm_set1_ps(first_weight);
    __m128 color_1 = _mm_mul_ps(c1fv, weight_1);

    __m128 weight_2 = _mm_set1_ps(second_weight);
    __m128 color_2 = _mm_mul_ps(c2fv, weight_2);

    __m128 color_new_vec = _mm_add_ps(color_1, color_2);
    __m128i color_new_int = _mm_cvttps_epi32(color_new_vec);

    /*int a[4];
    _mm_maskstore_epi32(a, mask, color_new_int);*/

    __m128i pack1 = _mm_packus_epi32(color_new_int, color_new_int);
    __m128i pack2 = _mm_packus_epi16(pack1, pack1);

    *(int*)new_color->rgba = _mm_extract_epi32(pack2, 0);
  }

  static void blendTwoColorsWithScaledWeights(const Color& first_color,
                                              FloatingPoint first_weight,
                                              const Color& second_color,
                                              FloatingPoint second_weight,
                                              Color* new_color) {
    __m128i c1v = _mm_setr_epi32(first_color.rgba[0], first_color.rgba[1],
                                 first_color.rgba[2], first_color.rgba[3]);
    __m128i c2v = _mm_setr_epi32(second_color.rgba[0], second_color.rgba[1],
                                 second_color.rgba[2], second_color.rgba[3]);

    __m128 c1fv = _mm_cvtepi32_ps(c1v);
    __m128 c2fv = _mm_cvtepi32_ps(c2v);

    __m128 weight_1 = _mm_set1_ps(first_weight);
    __m128 color_1 = _mm_mul_ps(c1fv, weight_1);

    __m128 weight_2 = _mm_set1_ps(second_weight);
    __m128 color_2 = _mm_mul_ps(c2fv, weight_2);

    __m128 color_new_vec = _mm_add_ps(color_1, color_2);
    __m128i color_new_int = _mm_cvttps_epi32(color_new_vec);

    /*int a[4];
    _mm_maskstore_epi32(a, mask, color_new_int);*/

    __m128i pack1 = _mm_packus_epi32(color_new_int, color_new_int);
    __m128i pack2 = _mm_packus_epi16(pack1, pack1);

    *(int*)new_color->rgba = _mm_extract_epi32(pack2, 0);
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
                                      const FloatingPoint grid_size_inv) {
  return AnyIndex(std::floor(point.x() * grid_size_inv),
                  std::floor(point.y() * grid_size_inv),
                  std::floor(point.z() * grid_size_inv));
}

// IMPORTANT NOTE: Due the limited accuracy of the FloatingPoint type, this
// function doesn't always compute the correct grid index for coordinates
// near the grid cell boundaries.
inline AnyIndex getGridIndexFromPoint(const Point& scaled_point) {
  return AnyIndex(std::floor(scaled_point.x()), std::floor(scaled_point.y()),
                  std::floor(scaled_point.z()));
}

inline AnyIndex getGridIndexFromOriginPoint(const Point& point,
                                            const FloatingPoint grid_size_inv) {
  return AnyIndex(std::round(point.x() * grid_size_inv),
                  std::round(point.y() * grid_size_inv),
                  std::round(point.z() * grid_size_inv));
}

inline Point getCenterPointFromGridIndex(const AnyIndex& idx,
                                         FloatingPoint grid_size) {
  return Point((static_cast<FloatingPoint>(idx.x()) + 0.5) * grid_size,
               (static_cast<FloatingPoint>(idx.y()) + 0.5) * grid_size,
               (static_cast<FloatingPoint>(idx.z()) + 0.5) * grid_size);
}

inline Point getOriginPointFromGridIndex(const AnyIndex& idx,
                                         FloatingPoint grid_size) {
  return Point(static_cast<FloatingPoint>(idx.x()) * grid_size,
               static_cast<FloatingPoint>(idx.y()) * grid_size,
               static_cast<FloatingPoint>(idx.z()) * grid_size);
}

inline BlockIndex getBlockIndexFromGlobalVoxelIndex(
    const AnyIndex& global_voxel_idx, FloatingPoint voxels_per_side_inv_) {
  __m128 voxels_per_side_inv = _mm_set1_ps(voxels_per_side_inv_);
  __m128i global_idx = _mm_set_epi32(0, global_voxel_idx(2),
                                     global_voxel_idx(1), global_voxel_idx(0));
  __m128 global_idx_float = _mm_cvtepi32_ps(global_idx);
  __m128 idx_over_voxels = _mm_mul_ps(global_idx_float, voxels_per_side_inv);
  __m128 floored = _mm_floor_ps(idx_over_voxels);
  __m128i floored_int = _mm_cvtps_epi32(floored);

  int ALIGNED_(16) result[4];
  _mm_store_si128((__m128i*)result, floored_int);
  return BlockIndex(result[0], result[1], result[2]);
}

inline VoxelIndex getLocalFromGlobalVoxelIndex(const AnyIndex& global_voxel_idx,
                                               int voxels_per_side) {
  DCHECK_EQ(voxels_per_side, 16);

  __m128i global_idx = _mm_set_epi32(0, global_voxel_idx(2),
                                     global_voxel_idx(1), global_voxel_idx(0));
  __m128i idx_abs = _mm_abs_epi32(global_idx);
  __m128i idx_by_16 = _mm_srli_epi32(idx_abs, 4);

  __m128i vec16 = _mm_set1_epi32(16);
  __m128i idx_round = _mm_mullo_epi32(idx_by_16, vec16);
  __m128i modulo = _mm_sub_epi32(idx_abs, idx_round);

  __m128i zeros = _mm_set1_epi32(0);
  __m128i ones = _mm_set1_epi32(1);

  __m128i mask_idx = _mm_cmplt_epi32(global_idx, zeros);
  // __m128i masked_mones = _mm_castps_si128(mask_idx);
  __m128i masked_ones = (__m128i)_mm_andnot_ps((__m128)mask_idx, (__m128)ones);
  __m128i correct = _mm_add_epi32(mask_idx, masked_ones);
  __m128i modulo_with_neg = _mm_mullo_epi32(modulo, correct);

  __m128i mask = _mm_cmplt_epi32(modulo_with_neg, zeros);

  __m128i additions = (__m128i)_mm_and_ps((__m128)mask, (__m128)vec16);
  __m128i modulo_in_range = _mm_add_epi32(modulo_with_neg, additions);

  int ALIGNED_(16) result[4];
  _mm_store_si128((__m128i*)result, modulo_in_range);
  return VoxelIndex(result[0], result[1], result[2]);
}

// Math functions.
inline int signum(FloatingPoint x) { return (x == 0) ? 0 : x < 0 ? -1 : 1; }

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
