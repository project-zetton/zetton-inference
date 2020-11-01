#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

#include "opencv2/opencv.hpp"
#include "ros/package.h"
#include "ros/ros.h"
#include "zetton_common/stream/gst_rtsp_stream.h"
#include "zetton_inference/detector/yolo_object_detector.h"

#define SHOW_GUI false

class RtspYoloObjectDetector {
 public:
  RtspYoloObjectDetector(std::string url, yolo_trt::Config config);

  bool open();
  bool isOpened();
  void release();
  bool read(cv::Mat& frame, zetton::inference::ObjectDetectionResults& result);
  bool isEmpty();

  void setProbThresh(float m_prob_thresh);
  void setWidthLimitation(float min_value, float max_value);
  void setHeightLimitation(float min_value, float max_value);

 private:
  void process();

  yolo_trt::Config config_;
  std::string url_;

  std::shared_ptr<zetton::inference::YoloObjectDetector> detector_;
  std::shared_ptr<zetton::common::GstRtspStream> streamer_;

  bool stop_flag_ = false;
  std::shared_ptr<std::thread> thread_;

  std::mutex mutex_;
  std::queue<std::pair<cv::Mat, zetton::inference::ObjectDetectionResults>>
      queue_;
};

RtspYoloObjectDetector::RtspYoloObjectDetector(std::string url,
                                               yolo_trt::Config config)
    : config_(std::move(config)), url_(std::move(url)) {
  open();
}

void RtspYoloObjectDetector::process() {
  cv::Mat frame;
  if (streamer_->read(frame)) {
    zetton::inference::ObjectDetectionResults results;
    detector_->Detect(frame, results);
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::make_pair(frame, results));
  }
}

bool RtspYoloObjectDetector::open() {
  // init detector
  detector_ = std::make_shared<zetton::inference::YoloObjectDetector>();
  detector_->Init(config_);
  // init streamer
  streamer_ = std::make_shared<zetton::common::GstRtspStream>();
  streamer_->open(url_, "avdec_h264");

  thread_ = std::make_shared<std::thread>([&]() {
    while (!stop_flag_) {
      process();
    }
  });

  return true;
}

bool RtspYoloObjectDetector::isOpened() { return streamer_->isOpened(); }

void RtspYoloObjectDetector::release() { stop_flag_ = true; }

bool RtspYoloObjectDetector::read(
    cv::Mat& frame, zetton::inference::ObjectDetectionResults& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pair = queue_.front();
  frame = pair.first;
  result = pair.second;
  queue_.pop();
  return true;
}

bool RtspYoloObjectDetector::isEmpty() { return queue_.empty(); }

void RtspYoloObjectDetector::setProbThresh(float m_prob_thresh) {
  detector_->SetProbThresh(m_prob_thresh);
}

void RtspYoloObjectDetector::setWidthLimitation(float min_value,
                                                float max_value) {
  detector_->SetWidthLimitation(min_value, max_value);
}

void RtspYoloObjectDetector::setHeightLimitation(float min_value,
                                                 float max_value) {
  detector_->SetHeightLimitation(min_value, max_value);
}

int main(int argc, char** argv) {
  // init ros
  ros::init(argc, argv, "example_rtsp_yolo_detector");

  // prepare yolo config
  yolo_trt::Config config_v4;
  std::string package_path = ros::package::getPath("zetton_inference");
  config_v4.net_type = yolo_trt::ModelType::YOLOV4;
  config_v4.file_model_cfg = package_path + "/asset/yolov4.cfg";
  config_v4.file_model_weights = package_path + "/asset/yolov4.weights";
  config_v4.inference_precison = yolo_trt::Precision::FP32;
  config_v4.detect_thresh = 0.4;

  // prepare stream url
  std::string rtsp_url =
      "rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen02.stream";
  std::string local_file = package_path + "/asset/demo.mp4";

  // init detector and streamer
  RtspYoloObjectDetector detector(rtsp_url, config_v4);
  detector.setProbThresh(0.4);
  detector.setWidthLimitation(50, 1920);
  detector.setHeightLimitation(50, 1920);

  // start
  ROS_INFO("Starting detection");
  while (detector.isOpened()) {
    cv::Mat frame;
    zetton::inference::ObjectDetectionResults results;
    if (!detector.isEmpty()) {
      // read frame from stream and detect objects by detector
      detector.read(frame, results);
      // print results
      for (const auto& result : results) {
        ROS_INFO_STREAM(result);
      }
      // show results in GUI
      if (SHOW_GUI) {
        for (auto& result : results) {
          result.Draw(frame);
        }
        cv::imshow("Results", frame);
        char key = cv::waitKey(10);
        if (key == 27) {
          ROS_INFO("Stopping detection");
          break;
        }
      }
    }
  }
  return 0;
}