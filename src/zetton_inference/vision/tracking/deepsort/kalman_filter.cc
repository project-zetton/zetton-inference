#include "zetton_inference/vision/tracking/deepsort/kalman_filter.h"

#include <cmath>

namespace zetton {
namespace inference {
namespace vision {
namespace deepsort {

int KalmanTracker::kf_count = 0;

void KalmanTracker::init_kf(std::array<float, 4> stateMat) {
  int stateNum = 7;
  int measureNum = 4;
  kf = cv::KalmanFilter(stateNum, measureNum, 0);

  measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

  kf.transitionMatrix =
      (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);

  setIdentity(kf.measurementMatrix);
  setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
  setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
  setIdentity(kf.errorCovPost, cv::Scalar::all(1));

  // initialize state vector with bounding box [x1,y1,x2,y2] in [cx,cy,s,r]
  // style
  kf.statePost.at<float>(0, 0) = (stateMat[0] + stateMat[2]) / 2;
  kf.statePost.at<float>(1, 0) = (stateMat[1] + stateMat[3]) / 2;
  kf.statePost.at<float>(2, 0) =
      (stateMat[2] - stateMat[0]) * (stateMat[3] - stateMat[1]);
  kf.statePost.at<float>(3, 0) =
      (stateMat[2] - stateMat[0]) / (stateMat[3] - stateMat[1]);
}

std::array<float, 4> KalmanTracker::Predict() {
  // predict
  cv::Mat p = kf.predict();
  m_age += 1;

  if (m_time_since_update > 0) m_hit_streak = 0;
  m_time_since_update += 1;

  std::array<float, 4> predictBox =
      GetBoxFromXYSR(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0),
                     p.at<float>(3, 0));

  m_history.push_back(predictBox);
  return m_history.back();
}

void KalmanTracker::Update(std::array<float, 4> stateMat, int classes,
                           float prob, const cv::Mat& feature) {
  m_time_since_update = 0;
  m_history.clear();
  m_hits += 1;
  m_hit_streak += 1;
  m_classes = classes;
  m_prob = prob;
  m_feature = feature.clone();

  // measurement
  measurement.at<float>(0, 0) = (stateMat[0] + stateMat[2]) / 2;
  measurement.at<float>(1, 0) = (stateMat[1] + stateMat[3]) / 2;
  measurement.at<float>(2, 0) =
      (stateMat[2] - stateMat[0]) * (stateMat[3] - stateMat[1]);
  measurement.at<float>(3, 0) =
      (stateMat[2] - stateMat[0]) / (stateMat[3] - stateMat[1]);

  // update
  kf.correct(measurement);
}

std::array<float, 4> KalmanTracker::GetState() {
  cv::Mat s = kf.statePost;
  return GetBoxFromXYSR(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0),
                        s.at<float>(3, 0));
}

std::array<float, 4> KalmanTracker::GetBoxFromXYSR(float cx, float cy, float s,
                                                   float r) {
  float w = std::sqrt(s * r);
  float h = s / w;
  float x = (cx - w / 2);
  float y = (cy - h / 2);

  if (x < 0 && cx > 0) {
    x = 0;
  }
  if (y < 0 && cy > 0) {
    y = 0;
  }

  return {x, y, x + w, y + h};
}

}  // namespace deepsort
}  // namespace vision
}  // namespace inference
}  // namespace zetton
