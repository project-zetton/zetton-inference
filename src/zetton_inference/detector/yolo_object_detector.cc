#include "zetton_inference/detector/yolo_object_detector.h"

#include <string>

#include "zetton_inference/base/object/object.h"

namespace zetton {
namespace inference {

std::string YoloObjectDetector::Name() const { return "YoloObjectDetector"; }

bool YoloObjectDetector::Init(const ObjectDetectorInitOptions& options) {
  return false;
}

bool YoloObjectDetector::Init(const yolo_trt::Config& config) {
  // copy config
  config_ = config;
  // init detector
  detector_ = std::make_shared<yolo_trt::Detector>();
  detector_->init(config_);
  return true;
}

bool YoloObjectDetector::Detect(const cv::Mat& frame,
                                std::vector<ObjectPtr>& results) {
  // note that the frame should be RGB order instead of BGR

  // construct inputs
  std::vector<cv::Mat> batch_frames;
  std::vector<yolo_trt::BatchResult> batch_results_raw;
  batch_frames.push_back(frame);
  // detect
  detector_->detect(batch_frames, batch_results_raw);
  // post-process
  FilterResults(batch_results_raw[0]);
  for (const auto& result : batch_results_raw[0]) {
    ObjectPtr obj;
    obj->camera_supplement.on_use = true;
    obj->camera_supplement.box.xmin = static_cast<float>(result.rect.tl().x);
    obj->camera_supplement.box.ymin = static_cast<float>(result.rect.tl().y);
    obj->camera_supplement.box.xmax = static_cast<float>(result.rect.br().x);
    obj->camera_supplement.box.ymax = static_cast<float>(result.rect.br().y);
    obj->type = result.id;
    obj->type_prob = result.prob;
    results.push_back(obj);
  }
  return true;
}

bool YoloObjectDetector::Detect(
    const std::vector<cv::Mat>& batch_frames,
    std::vector<std::vector<ObjectPtr>>& batch_results) {
  // detect
  std::vector<yolo_trt::BatchResult> batch_results_raw;
  detector_->detect(batch_frames, batch_results_raw);
  // post-process
  for (auto& batch_result_raw : batch_results_raw) {
    FilterResults(batch_result_raw);
    std::vector<ObjectPtr> batch_result;
    for (const auto& result : batch_result_raw) {
      ObjectPtr obj;
      obj->camera_supplement.on_use = true;
      obj->camera_supplement.box.xmin = static_cast<float>(result.rect.tl().x);
      obj->camera_supplement.box.ymin = static_cast<float>(result.rect.tl().y);
      obj->camera_supplement.box.xmax = static_cast<float>(result.rect.br().x);
      obj->camera_supplement.box.ymax = static_cast<float>(result.rect.br().y);
      obj->type = result.id;
      obj->type_prob = result.prob;
      batch_result.push_back(obj);
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
