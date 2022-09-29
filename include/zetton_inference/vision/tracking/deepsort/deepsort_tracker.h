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

/// \brief parameters for deepsort tracker
struct DeepSORTTrackerParams {
  /// \brief input image size
  /// \note the input image will be resized to this size
  /// \details tuple of (width, height)
  std::vector<int> size = {128, 256};
  /// \brief maximum number of objects to be tracked
  int max_age = 70;
  /// \brief threshold for the IoU value to determine whether two boxes are
  /// matched
  float iou_threshold = 0.3;
  /// \brief threshold for the cosine distance to determine whether two features
  /// are matched
  float sim_threshold = 0.4;
  /// \brief whether to distinguish different classes in tracking
  bool agnostic = true;
};

/// \brief an object tracking algorithm based on deep appearance features and
/// Kalman filtering
class DeepSORTTracker {
 public:
  /// \brief update the tracker with detection and reid results
  void Update(const DetectionResult &detections, const ReIDResult &features);

 public:
  /// \brief current tracking results
  TrackingResult tracking_results;
  /// \brief parameters for deepsort tracker
  DeepSORTTrackerParams params;

 private:
  /// \brief calculate the IoU value between two bounding boxes
  /// \param det_a first bounding box
  /// \param det_b second bounding box
  static float IOUCalculate(const deepsort::KalmanTracker::StateType &det_a,
                            const deepsort::KalmanTracker::StateType &det_b);
  /// \brief match detections and trackers by given cost matrix
  /// \param mat cost matrix
  /// \param unassigned_detections unassigned detections
  /// \param unassigned_trackers unassigned trackers
  /// \param assigned_pairs assigned detection-tracker pairs
  /// \param det_num number of detections
  /// \param trk_num number of trackers
  /// \param b_iou matching threshold
  void Alignment(std::vector<std::vector<double>> mat,
                 std::set<int> &unassigned_detections,
                 std::set<int> &unassigned_tracks,
                 std::vector<cv::Point> &assigned_pairs, int det_num,
                 int trk_num, bool b_iou);
  /// \brief match detections and trackers by IoU
  /// \param predict_boxes predicted bounding boxes from Kalman tracker
  /// \param unassigned_detections unassigned detections
  /// \param unassigned_trackers unassigned trackers
  /// \param assigned_pairs assigned detection-tracker pairs
  void IOUMatching(const TrackingResult &predict_boxes,
                   std::set<int> &unassigned_detections,
                   std::set<int> &unassigned_tracks,
                   std::vector<cv::Point> &assigned_pairs);
  /// \brief match detections and trackers by extracted features
  /// \param predict_boxes predicted bounding boxes from Kalman tracker
  /// \param unassigned_detections unassigned detections
  /// \param unassigned_trackers unassigned trackers
  /// \param assigned_pairs assigned detection-tracker pairs
  void FeatureMatching(const TrackingResult &predict_boxes,
                       std::set<int> &unassigned_detections,
                       std::set<int> &unassigned_tracks,
                       std::vector<cv::Point> &assigned_pairs);

 private:
  /// \brief current trackers
  std::vector<deepsort::KalmanTracker> kalman_boxes_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
