#pragma once

#include <opencv2/core.hpp>

#include "zetton_inference/vision/common/result.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Utility class for result visualization
class Visualization {
 public:
  /// \brief get the color map for visualization
  /// \param num_classes number of classes
  static const std::vector<int>& GetColorMap(int num_classes = 1000);

 public:
  /// \brief draw the detection result on image
  /// \details only support visualize num_classes <= 1000 by default. If need to
  /// visualize num_classes > 1000, please call GetColorMap(num_classes) first
  static cv::Mat Visualize(const cv::Mat& im, const DetectionResult& result,
                           float score_threshold = 0.0, int line_size = 1,
                           float font_size = 0.5f);

 public:
  /// \brief number of colors for visualization
  static int num_classes_;
  /// \brief color map for visualization
  static std::vector<int> color_map_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
