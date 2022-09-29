#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace zetton {
namespace inference {
namespace vision {
namespace deepsort {

// This class represents the internel state of individual tracked objects
// observed as bounding box.
class KalmanTracker {
 public:
  /// \brief constructor
  KalmanTracker() {
    init_kf(std::array<float, 4>());
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
    m_classes = -1;
    m_prob = 0.0;
    // kf_count++;
  }

  /// \brief constructor with bounding box
  KalmanTracker(std::array<float, 4> initRect, int classes, float prob) {
    init_kf(initRect);
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
    kf_count++;
    m_classes = classes;
    m_prob = prob;
  }

  /// \brief destructor
  ~KalmanTracker() { m_history.clear(); }

  /// \brief predict the estimated bounding box
  std::array<float, 4> Predict();

  /// \brief update the state vector with observed bounding box
  void Update(std::array<float, 4> stateMat, int classes, float prob,
              const cv::Mat& feature);

 public:
  /// \brief get the current state vector
  std::array<float, 4> GetState();

  /// \brief convert bounding box from [cx,cy,s,r] to [x,y,w,h] style
  std::array<float, 4> GetBoxFromXYSR(float cx, float cy, float s, float r);

 public:
  static int kf_count;

  int m_time_since_update;
  int m_hits;
  int m_hit_streak;
  int m_age;
  int m_id;
  int m_classes;
  float m_prob;
  cv::Mat m_feature;

 private:
  void init_kf(std::array<float, 4> stateMat);

  cv::KalmanFilter kf;
  cv::Mat measurement;

  std::vector<std::array<float, 4>> m_history;
};

}  // namespace deepsort
}  // namespace vision
}  // namespace inference
}  // namespace zetton
