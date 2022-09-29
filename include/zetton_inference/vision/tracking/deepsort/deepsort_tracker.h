#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <set>
#include <vector>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/tracking/deepsort/kalman_filter.h"

namespace zetton {
namespace inference {
namespace vision {

struct DeepSORTTrackerParams {
  /// \brief input image size
  /// \note the input image will be resized to this size
  /// \details tuple of (width, height)
  std::vector<int> size = {128, 256};
  int max_age = 70;
  float iou_threshold = 0.3;
  float sim_threshold = 0.4;
  bool agnostic = true;
};

class DeepSORTTracker {
 public:
  void Update(const DetectionResult &detections, const ReIDResult &features);

 public:
  TrackingResult tracking_results;
  DeepSORTTrackerParams params;

 private:
  static float IOUCalculate(const deepsort::KalmanTracker::StateType &det_a,
                            const deepsort::KalmanTracker::StateType &det_b);
  void Alignment(std::vector<std::vector<double>> mat,
                 std::set<int> &unmatchedDetections,
                 std::set<int> &unmatchedTrajectories,
                 std::vector<cv::Point> &matchedPairs, int det_num, int trk_num,
                 bool b_iou);
  void IOUMatching(const TrackingResult &predict_boxes,
                   std::set<int> &unmatchedDetections,
                   std::set<int> &unmatchedTrajectories,
                   std::vector<cv::Point> &matchedPairs);
  void FeatureMatching(const TrackingResult &predict_boxes,
                       std::set<int> &unmatchedDetections,
                       std::set<int> &unmatchedTrajectories,
                       std::vector<cv::Point> &matchedPairs);

 private:
  std::vector<deepsort::KalmanTracker> kalman_boxes_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
