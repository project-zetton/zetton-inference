#pragma once

#include <limits>
#include <memory>
#include <vector>

namespace zetton {
namespace inference {

template <typename T>
struct alignas(16) Point {
  T x = 0;
  T y = 0;
  T z = 0;
  T intensity = 0;
  using Type = T;
};

template <typename T>
struct PointXYZIT : public Point<T> {
  double timestamp = 0.0;
};

template <typename T>
struct PointXYZITH : public PointXYZIT<T> {
  float height = std::numeric_limits<float>::max();
};

template <typename T>
struct PointXYZITHB : public PointXYZITH<T> {
  int32_t beam_id = -1;
};

template <typename T>
struct PointXYZITHBL : public PointXYZITHB<T> {
  uint8_t label = 0;
};

using PointF = Point<float>;
using PointD = Point<double>;

using PointXYZIF = Point<float>;
using PointXYZID = Point<double>;
using PointXYZITF = PointXYZIT<float>;
using PointXYZITD = PointXYZIT<double>;
using PointXYZITHF = PointXYZITH<float>;
using PointXYZITHD = PointXYZITH<double>;
using PointXYZITHBF = PointXYZITHB<float>;
using PointXYZITHBD = PointXYZITHB<double>;
using PointXYZITHBLF = PointXYZITHBL<float>;
using PointXYZITHBLD = PointXYZITHBL<double>;

const std::size_t kDefaultReservePointNum = 50000;

struct PointIndices {
  PointIndices() { indices.reserve(kDefaultReservePointNum); }

  std::vector<int> indices;

  using Ptr = std::shared_ptr<PointIndices>;
  using ConstPtr = std::shared_ptr<const PointIndices>;
};

template <typename T>
struct Point2D {
  T x = 0;
  T y = 0;
};

using Point2DF = Point2D<float>;
using Point2DI = Point2D<int>;
using Point2DD = Point2D<double>;

template <typename T>
struct Point3D {
  T x = 0;
  T y = 0;
  T z = 0;
};

using Point3DF = Point3D<float>;
using Point3DI = Point3D<int>;
using Point3DD = Point3D<double>;

}  // namespace inference
}  // namespace zetton
