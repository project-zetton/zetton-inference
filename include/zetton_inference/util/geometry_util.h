#pragma once

#include <opencv2/opencv.hpp>

#include "zetton_inference/base/geometry/box.h"

namespace zetton {
namespace inference {

template <typename T>
cv::Rect GetCvRect(const BBox2D<T> &box) {
  return {cv::Point(static_cast<int>(box.xmin), static_cast<int>(box.ymin)),
          cv::Point(static_cast<int>(box.xmax), static_cast<int>(box.ymax))};
}

}  // namespace inference
}  // namespace zetton
