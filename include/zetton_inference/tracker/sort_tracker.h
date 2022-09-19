#pragma once

#include "zetton_inference/base/object/object.h"
#include "zetton_inference/interface/base_object_detector.h"
#include "zetton_inference/interface/base_object_tracker.h"
#include "zetton_inference/tracker/sort/hungarian.h"
#include "zetton_inference/tracker/sort/kalman_filter_cv.h"

namespace zetton {
namespace inference {

class SortTracker : public BaseObjectTracker {
 public:
  SortTracker() = default;
  ~SortTracker() override = default;

  std::string Name() const final;

  bool Init(const ObjectTrackerInitOptions &options =
                ObjectTrackerInitOptions()) override;

  bool Track() override { return true; };
  bool Track(const cv::Mat &frame, const double &timestamp,
             std::vector<ObjectPtr> &detections);

  std::vector<ObjectPtr> &tracks() { return tracking_results; }

 private:
  // Computes IOU between two bounding boxes
  inline double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON) return 0;

    return (double)(in / un);
  }

 private:
  int frame_count_ = 0;
  int max_age_ = 10;
  int active_age = 3;
  int min_hits_ = 3;
  double iou_threshold_ = 0.3;
  double duplicated_threshold_ = 0.8;
  std::vector<tracker::sort::KalmanTracker> trackers_;

  std::vector<cv::Rect_<float>> predicted_boxes;
  std::vector<std::vector<double>> iou_matrix;
  std::vector<int> assignments;
  std::set<int> unmatched_detections;
  std::set<int> unmatched_tracks;
  std::set<int> all_items;
  std::set<int> matched_items;
  std::vector<cv::Point> matched_pairs;
  std::set<int> updated_tracks;
  std::vector<int> duplicated_tracks;
  std::vector<ObjectPtr> tracking_results;

  unsigned int num_tracks = 0;
  unsigned int num_detections = 0;
};

}  // namespace inference
}  // namespace zetton
