#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

namespace zetton {
namespace inference {
namespace tracker {
namespace sort {

using namespace std;
using namespace cv;

#define StateType Rect_<float>

// This class represents the internel state of individual tracked objects
// observed as bounding box.
class KalmanTracker {
 public:
  KalmanTracker(const int& frame_id) {
    init_kf(StateType());
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
    m_start_frame = frame_id;
    // kf_count++;
  }
  KalmanTracker(const int& frame_id, StateType initRect) {
    init_kf(initRect);
    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = kf_count;
    m_start_frame = frame_id;
    kf_count++;
  }

  ~KalmanTracker() { m_history.clear(); }

  StateType predict();
  void update(StateType stateMat);

  StateType get_state();
  StateType get_rect_xysr(float cx, float cy, float s, float r);

  static int kf_count;

  int m_time_since_update;
  int m_hits;
  int m_hit_streak;
  int m_age;
  int m_id;
  int m_start_frame;

 private:
  void init_kf(StateType stateMat);

  cv::KalmanFilter kf;
  cv::Mat measurement;

  std::vector<StateType> m_history;
};

}  // namespace sort
}  // namespace tracker
}  // namespace inference
}  // namespace zetton
