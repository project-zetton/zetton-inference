#include "zetton_inference/vision/util/visualize.h"

#include <fmt/format.h>

#include <opencv2/imgproc.hpp>

namespace zetton {
namespace inference {
namespace vision {

int Visualization::num_classes_ = 0;
std::vector<int> Visualization::color_map_ = std::vector<int>();

const std::vector<int>& Visualization::GetColorMap(int num_classes) {
  if (num_classes < num_classes_) {
    return color_map_;
  }
  num_classes_ = num_classes;
  std::vector<int>().swap(color_map_);
  color_map_.resize(3 * num_classes_, 0);
  for (int i = 0; i < num_classes_; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      color_map_[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      color_map_[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      color_map_[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return color_map_;
}

cv::Mat Visualization::Visualize(const cv::Mat& im,
                                 const DetectionResult& result,
                                 float score_threshold, int line_size,
                                 float font_size) {
  auto color_map = GetColorMap();
  auto vis_im = im.clone();
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    // check score
    if (result.scores[i] < score_threshold) {
      continue;
    }

    // draw rectangle
    int x1 = static_cast<int>(result.boxes[i][0]);
    int y1 = static_cast<int>(result.boxes[i][1]);
    int x2 = static_cast<int>(result.boxes[i][2]);
    int y2 = static_cast<int>(result.boxes[i][3]);
    int box_h = y2 - y1;
    int box_w = x2 - x1;
    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    cv::Rect rect(x1, y1, box_w, box_h);
    cv::rectangle(vis_im, rect, rect_color, line_size);

    // draw text
    std::string text =
        fmt::format("{},{:.2f}", result.label_ids[i], result.scores[i]);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Point origin;
    origin.x = x1;
    origin.y = y1;
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
  }
  return vis_im;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
