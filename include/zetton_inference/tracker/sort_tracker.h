#pragma once

#include "zetton_inference/interface/base_object_detector.h"
#include "zetton_inference/interface/base_object_tracker.h"
#include "zetton_inference/tracker/sort/hungarian.h"
#include "zetton_inference/tracker/sort/kalman_filter_cv.h"

namespace zetton {
namespace inference {

class SortTracker : public BaseObjectTracker {
  typedef struct TrackingBox {
    int frame;
    int id;
    cv::Rect_<float> box;

    inline void Draw(cv::Mat &frame) {
      cv::rectangle(frame, box, cv::Scalar(255, 255, 0), 2);
      std::stringstream stream;
      stream << "Track " << id;
      cv::putText(frame, stream.str(), cv::Point(box.x, box.y - 5), 0, 0.5,
                  cv::Scalar(255, 255, 0), 2);
    }

    inline friend std::ostream &operator<<(std::ostream &os,
                                           const TrackingBox &track) {
      os << track.id << ":" << track.box;
      return os;
    }
  } TrackingBox;

 public:
  SortTracker() = default;
  ~SortTracker() override = default;

  bool Init() override;
  void Infer() override{};

  bool Track() override { return true; };
  bool Track(const cv::Mat &frame, const ros::Time &timestamp,
             const ObjectDetectionResults &detections);

  std::vector<TrackingBox> &tracks() { return frameTrackingResult; }

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
  int max_age_ = 6;
  int active_age = 2;
  int min_hits_ = 3;
  double iouThreshold_ = 0.3;
  std::vector<tracker::sort::KalmanTracker> trackers_;

  std::vector<cv::Rect_<float>> predictedBoxes;
  std::vector<std::vector<double>> iouMatrix;
  std::vector<int> assignment;
  std::set<int> unmatchedDetections;
  std::set<int> unmatchedTrajectories;
  std::set<int> allItems;
  std::set<int> matchedItems;
  std::vector<cv::Point> matchedPairs;
  std::vector<TrackingBox> frameTrackingResult;
  unsigned int trkNum = 0;
  unsigned int detNum = 0;
};

}  // namespace inference
}  // namespace zetton