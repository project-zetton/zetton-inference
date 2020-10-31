#include "zetton_inference/detector/yolo_object_detector.h"

#include <ros/package.h>

#include <string>

#include "zetton_common/util/ros_util.h"

namespace zetton {
namespace inference {

bool YoloObjectDetector::Init() {
  // get params
  auto nh_ = zetton::common::RosNodeHandler::Instance()->GetNh();
  XmlRpc::XmlRpcValue params;
  GPARAM("/zetton_inference/yolo_object_detector", params);
  std::string package_path = ros::package::getPath("zetton_inference");

  auto net_type = static_cast<std::string>(params["net_type"]);
  if (net_type == "YOLOV4") {
    config_.net_type = yolo_trt::ModelType::YOLOV4;
  } else if (net_type == "YOLOV4_TINY") {
    config_.net_type = yolo_trt::ModelType::YOLOV4_TINY;
  } else {
    ROS_ERROR_STREAM("Unsupported net type: " << net_type);
    return false;
  }
  auto precision = static_cast<std::string>(params["inference_precision"]);
  if (precision == "FP32") {
    config_.inference_precison = yolo_trt::Precision::FP32;
  } else if (precision == "FP16") {
    config_.inference_precison = yolo_trt::Precision::FP16;
  } else if (precision == "INT8") {
    config_.inference_precison = yolo_trt::Precision::INT8;
    config_.calibration_image_list_file_txt =
        package_path +
        static_cast<std::string>(params["calibration_image_list_file_txt"]);
  } else {
    ROS_ERROR_STREAM("Unsupported inference precision: " << precision);
    return false;
  }

  config_.file_model_cfg =
      package_path + static_cast<std::string>(params["model_cfg"]);
  config_.file_model_weights =
      package_path + static_cast<std::string>(params["model_weights"]);
  config_.gpu_id = static_cast<int>(params["gpu_id"]);
  config_.detect_thresh =
      static_cast<float>(static_cast<double>(params["detect_thresh"]));
  config_.nms_thresh =
      static_cast<float>(static_cast<double>(params["nms_thresh"]));
  min_width_ = static_cast<float>(static_cast<int>(params["min_width"]));
  max_width_ = static_cast<float>(static_cast<int>(params["max_width"]));
  min_height_ = static_cast<float>(static_cast<int>(params["min_height"]));
  max_height_ = static_cast<float>(static_cast<int>(params["max_height"]));

  // init detector
  detector_ = std::make_shared<yolo_trt::Detector>();
  detector_->init(config_);

  return true;
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

}  // namespace inference
}  // namespace zetton
