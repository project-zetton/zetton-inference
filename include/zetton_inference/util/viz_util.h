#pragma once

#include <fmt/format.h>

#include <opencv2/opencv.hpp>

#include "zetton_inference/base/object/object.h"
#include "zetton_inference/util/geometry_util.h"

namespace zetton {
namespace inference {

inline void DrawBoundingBoxOnCvImage(cv::Mat& frame, const ObjectPtr& object) {
  auto bbox = GetCvRect(object->camera_supplement.box);
  cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
  std::stringstream stream;
  cv::putText(frame, fmt::format("{}@{}", object->type, object->type_prob),
              cv::Point(bbox.x, bbox.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
}

}  // namespace inference
}  // namespace zetton
