#include "zetton_inference/detector/yolo_object_detector.h"

#include <string>

namespace zetton {
namespace inference {

bool YoloObjectDetector::Init() { return false; }

bool YoloObjectDetector::Init(const yolo_trt::Config& config) {
  // copy config
  config_ = config;
  // init detector
  detector_ = std::make_shared<yolo_trt::Detector>();
  detector_->init(config_);
  return true;
}

bool YoloObjectDetector::Detect(const cv::Mat& frame,
                                ObjectDetectionResults& results) {
  // note that the frame should be RGB order instead of BGR

  // construct inputs
  std::vector<cv::Mat> batch_frames;
  std::vector<yolo_trt::BatchResult> batch_results_raw;
  batch_frames.push_back(frame);
  // detect
  detector_->detect(batch_frames, batch_results_raw);
  // post-process
  FilterResults(batch_results_raw[0]);
  for (const auto& res : batch_results_raw[0]) {
    results.emplace_back(res.rect, res.id, res.prob);
  }
  return true;
}

bool YoloObjectDetector::Detect(const std::vector<cv::Mat>& batch_frames,
                                BatchObjectDetectionResults& batch_results) {
  // detect
  std::vector<yolo_trt::BatchResult> batch_results_raw;
  detector_->detect(batch_frames, batch_results_raw);
  // post-process
  for (auto& batch_result_raw : batch_results_raw) {
    FilterResults(batch_result_raw);
    ObjectDetectionResults batch_result;
    for (const auto& result : batch_result_raw) {
      batch_result.emplace_back(result.rect, result.id, result.prob);
    }
    batch_results.push_back(batch_result);
  }
  return true;
}

void YoloObjectDetector::SetProbThresh(float m_prob_thresh) {
  detector_->setProbThresh(m_prob_thresh);
}

void YoloObjectDetector::SetWidthLimitation(float min_value, float max_value) {
  min_width_ = min_value;
  max_width_ = max_value;
}

void YoloObjectDetector::SetHeightLimitation(float min_value, float max_value) {
  min_height_ = min_value;
  max_height_ = max_value;
}

void YoloObjectDetector::FilterResults(yolo_trt::BatchResult& results) {
  auto end = std::remove_if(
      results.begin(), results.end(), [&](const yolo_trt::Result& result) {
        bool is_max_width_valid =
            max_width_ >= 0 && result.rect.width <= max_width_;
        bool is_min_width_valid =
            min_width_ >= 0 && result.rect.width >= min_width_;
        bool is_max_height_valid =
            max_height_ >= 0 && result.rect.width <= max_height_;
        bool is_min_height_valid =
            min_height_ >= 0 && result.rect.width >= min_height_;
        return !(is_max_width_valid && is_min_width_valid &&
                 is_max_height_valid && is_min_height_valid);
      });
  results.erase(end, results.end());
}

ZETTON_REGISTER_OBJECT_DETECTOR(YoloObjectDetector)

}  // namespace inference
}  // namespace zetton
