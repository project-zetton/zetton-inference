#pragma once

#include "yolo_trt/class_detector.h"
#include "zetton_inference/interface/base_object_detector.h"

namespace zetton {
namespace inference {

class YoloObjectDetector : public BaseObjectDetector {
 public:
  YoloObjectDetector() = default;
  ~YoloObjectDetector() override = default;

  std::string Name() const final;

  bool Init(const ObjectDetectorInitOptions& options =
                ObjectDetectorInitOptions()) final;
  bool Init(const yolo_trt::Config& config);

  bool Detect(const cv::Mat& frame, std::vector<ObjectPtr>& results) final;
  bool Detect(const std::vector<cv::Mat>& batch_frames,
              std::vector<std::vector<ObjectPtr>>& batch_results);

 public:
  void SetProbThresh(float m_prob_thresh);
  void SetWidthLimitation(float min_value, float max_value);
  void SetHeightLimitation(float min_value, float max_value);

 private:
  void FilterResults(yolo_trt::BatchResult& results);

  yolo_trt::Config config_;

  int min_width_ = -1;
  int max_width_ = -1;
  int min_height_ = -1;
  int max_height_ = -1;

  std::shared_ptr<yolo_trt::Detector> detector_;
};

}  // namespace inference
}  // namespace zetton
