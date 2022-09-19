#pragma once

#include <fmt/format.h>

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "zetton_inference/base/object/object.h"
#include "zetton_inference/util/geometry_util.h"

namespace zetton {
namespace inference {

struct VisualizationOptions {
  cv::Scalar background_color = {255, 0, 0};
  cv::Scalar text_color = {0, 0, 255};
};

inline void DrawBoundingBoxOnCvImage(
    cv::Mat& frame, const ObjectPtr& object,
    const VisualizationOptions& options = VisualizationOptions()) {
  auto bbox = GetCvRect(object->camera_supplement.box);
  cv::rectangle(frame, bbox, options.background_color, 2);
  std::stringstream stream;
  cv::putText(frame, fmt::format("{}@{}", object->type, object->type_prob),
              cv::Point(bbox.x, bbox.y - 5), 0, 0.5, options.text_color, 2);
}

}  // namespace inference
}  // namespace zetton
