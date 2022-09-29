#include "zetton_inference/vision/tracking/deepsort/kalman_filter.h"

#include <cmath>

#include "zetton_inference/vision/util/geometry.h"

namespace zetton {
namespace inference {
namespace vision {
namespace deepsort {

int KalmanTracker::kf_count = 0;

KalmanTracker::KalmanTracker() {
  Init(StateType());
  time_since_update = 0;
  hits = 0;
  hit_streak = 0;
  age = 0;
  id = kf_count;
  classes = -1;
  prob = 0.0;
  // kf_count++;
}

KalmanTracker::KalmanTracker(StateType initRect, int classes, float prob) {
  Init(initRect);
  time_since_update = 0;
  hits = 0;
  hit_streak = 0;
  age = 0;
  id = kf_count;
  kf_count++;
  classes = classes;
  prob = prob;
}

void KalmanTracker::Init(StateType stateMat) {
  int stateNum = 7;
  int measureNum = 4;
  kf_ = cv::KalmanFilter(stateNum, measureNum, 0);

  measurement_ = cv::Mat::zeros(measureNum, 1, CV_32F);

  kf_.transitionMatrix =
      (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);

  setIdentity(kf_.measurementMatrix);
  setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
  setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
  setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

  // initialize state vector with bounding box [x1,y1,x2,y2] in [cx,cy,s,r]
  // style
  kf_.statePost.at<float>(0, 0) = (stateMat[0] + stateMat[2]) / 2;
  kf_.statePost.at<float>(1, 0) = (stateMat[1] + stateMat[3]) / 2;
  kf_.statePost.at<float>(2, 0) =
      (stateMat[2] - stateMat[0]) * (stateMat[3] - stateMat[1]);
  kf_.statePost.at<float>(3, 0) =
      (stateMat[2] - stateMat[0]) / (stateMat[3] - stateMat[1]);
}

KalmanTracker::StateType KalmanTracker::Predict() {
  // predict
  cv::Mat p = kf_.predict();
  age += 1;

  if (time_since_update > 0) hit_streak = 0;
  time_since_update += 1;

  StateType predictBox =
      GetTLBRFromXYSR({p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0),
                       p.at<float>(3, 0)});

  history_.push_back(predictBox);
  return history_.back();
}

void KalmanTracker::Update(StateType stateMat, int classes, float prob,
                           const cv::Mat& feature) {
  time_since_update = 0;
  history_.clear();
  hits += 1;
  hit_streak += 1;
  classes = classes;
  prob = prob;
  feature = feature.clone();

  // measurement
  StateType stateMatXYSR = GetXYSRFromTLBR(stateMat);
  measurement_.at<float>(0, 0) = stateMatXYSR[0];
  measurement_.at<float>(1, 0) = stateMatXYSR[1];
  measurement_.at<float>(2, 0) = stateMatXYSR[2];
  measurement_.at<float>(3, 0) = stateMatXYSR[3];

  // update
  kf_.correct(measurement_);
}

KalmanTracker::StateType KalmanTracker::GetState() {
  cv::Mat s = kf_.statePost;
  return GetTLBRFromXYSR({s.at<float>(0, 0), s.at<float>(1, 0),
                          s.at<float>(2, 0), s.at<float>(3, 0)});
}

}  // namespace deepsort
}  // namespace vision
}  // namespace inference
}  // namespace zetton
