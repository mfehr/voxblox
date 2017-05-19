#ifndef VOXBLOX_FAST_CORE_COLOR_H_
#define VOXBLOX_FAST_CORE_COLOR_H_

#include "voxblox_fast/core/common.h"

namespace voxblox_fast {

// Color maps.

// The input h is on a scale between 0.0 and 1.0.
inline Color rainbowColorMap(double h) {
  // Directly from OctomapProvider in octomap.
  Color color;
  color.rgba[3] = 255;
  // blend over HSV-values (more colors)

  double s = 1.0;
  double v = 1.0;

  h -= floor(h);
  h *= 6;
  int i;
  double m, n, f;

  i = floor(h);
  f = h - i;
  if (!(i & 1)) f = 1 - f;  // if i is even
  m = v * (1 - s);
  n = v * (1 - s * f);

  switch (i) {
    case 6:
    case 0:
      color.rgba[0] = 255 * v;
      color.rgba[1] = 255 * n;
      color.rgba[2] = 255 * m;
      break;
    case 1:
      color.rgba[0] = 255 * n;
      color.rgba[1] = 255 * v;
      color.rgba[2] = 255 * m;
      break;
    case 2:
      color.rgba[0] = 255 * m;
      color.rgba[1] = 255 * v;
      color.rgba[2] = 255 * n;
      break;
    case 3:
      color.rgba[0] = 255 * m;
      color.rgba[1] = 255 * n;
      color.rgba[2] = 255 * v;
      break;
    case 4:
      color.rgba[0] = 255 * n;
      color.rgba[1] = 255 * m;
      color.rgba[2] = 255 * v;
      break;
    case 5:
      color.rgba[0] = 255 * v;
      color.rgba[1] = 255 * m;
      color.rgba[2] = 255 * n;
      break;
    default:
      color.rgba[0] = 255;
      color.rgba[1] = 127;
      color.rgba[2] = 127;
      break;
  }

  return color;
}

inline Color grayColorMap(double h) {
  Color color;
  color.rgba[3] = 255;

  color.rgba[0] = round(h * 255);
  color.rgba[1] = 128;
  color.rgba[2] = color.rgba[0];

  return color;
}

inline Color randomColor() {
  Color color;

  color.rgba[3] = 255;

  color.rgba[0] = rand() % 256;
  color.rgba[1] = rand() % 256;
  color.rgba[2] = rand() % 256;

  return color;
}

}  // namespace voxblox

#endif  // VOXBLOX_FAST_CORE_COLOR_H_
