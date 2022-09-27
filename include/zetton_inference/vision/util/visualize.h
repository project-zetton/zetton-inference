#pragma once

#include <opencv2/core.hpp>

#include "zetton_inference/vision/base/result.h"

namespace zetton {
namespace inference {
namespace vision {

class Visualization {
 public:
  static const std::vector<int>& GetColorMap(int num_classes = 1000);

 public:
  /// \brief
  /// \details only support visualize num_classes <= 1000 by default. If need to
  /// visualize num_classes > 1000, please call GetColorMap(num_classes) first
  static cv::Mat Visualize(const cv::Mat& im, const DetectionResult& result,
                           float score_threshold = 0.0, int line_size = 1,
                           float font_size = 0.5f);

 public:
  static int num_classes_;
  static std::vector<int> color_map_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
