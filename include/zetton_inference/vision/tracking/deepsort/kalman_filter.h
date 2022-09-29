#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace zetton {
namespace inference {
namespace vision {
namespace deepsort {

/// \brief This class represents the internel state of individual tracked
/// objects observed as bounding box.
class KalmanTracker {
 public:
  using StateType = std::array<float, 4>;

  /// \brief constructor
  KalmanTracker();

  /// \brief constructor with bounding box
  /// \param initRect bounding box in TLBR format
  /// \param classes label id
  KalmanTracker(StateType initRect, int classes, float prob);

  /// \brief destructor
  ~KalmanTracker() { history_.clear(); }

 public:
  /// \brief predict the estimated bounding box
  StateType Predict();

  /// \brief update the state vector with observed bounding box
  void Update(StateType stateMat, int classes, float prob,
              const cv::Mat& feature);

 public:
  /// \brief get the current state vector (in TLBR format)
  StateType GetState();

 public:
  static int kf_count;

  int time_since_update;
  int hits;
  int hit_streak;
  int age;
  int id;
  int classes;
  float prob;
  cv::Mat feature;

 private:
  void Init(StateType stateMat);

 private:
  cv::KalmanFilter kf_;
  cv::Mat measurement_;
  std::vector<StateType> history_;
};

}  // namespace deepsort
}  // namespace vision
}  // namespace inference
}  // namespace zetton
