#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace zetton {
namespace inference {
namespace vision {
namespace deepsort {

struct KalmanTrackerData {
  int time_since_update;
  int hits;
  int hit_streak;
  int age;
  int id;
  int label_id;
  float score;
  cv::Mat feature;
};

/// \brief This class represents the internel state of individual tracked
/// objects observed as bounding box.
class KalmanTracker {
 public:
  using StateType = std::array<float, 4>;

  /// \brief constructor
  KalmanTracker();

  /// \brief constructor with bounding box
  /// \param box bounding box in TLBR format
  /// \param label_id label id
  /// \param prob detection probability
  KalmanTracker(const StateType& box, int label_id, float score);

  /// \brief destructor
  ~KalmanTracker() { history_.clear(); }

 public:
  /// \brief predict the estimated bounding box
  StateType Predict();

  /// \brief update the state vector with observed bounding box
  /// \param box bounding box in TLBR format
  /// \param label_id label id
  /// \param prob detection probability
  /// \param feature feature vector
  void Update(const StateType& box, int label_id, float score,
              const cv::Mat& feature);

 public:
  /// \brief get the current state vector (in TLBR format)
  StateType GetState();

 public:
  /// \brief total number of instances, used for tracking id
  static int kf_count;
  /// \brief Stored data for tracker
  KalmanTrackerData data;

 private:
  /// \brief initialize Kalman filter
  void Init(StateType box);

 private:
  /// \brief Kalman filter from OpenCV
  cv::KalmanFilter kf_;
  /// \brief measurement matrix
  cv::Mat measurement_;
  /// \brief history of tracked bounding box
  std::vector<StateType> history_;
};

}  // namespace deepsort
}  // namespace vision
}  // namespace inference
}  // namespace zetton
