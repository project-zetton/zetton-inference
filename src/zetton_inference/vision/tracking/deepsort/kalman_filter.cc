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
  data.time_since_update = 0;
  data.hits = 0;
  data.hit_streak = 0;
  data.age = 0;
  data.id = kf_count;
  data.label_id = -1;
  data.score = 0.0;
  // kf_count++;
}

KalmanTracker::KalmanTracker(const StateType& box, int label_id, float score) {
  Init(box);
  data.time_since_update = 0;
  data.hits = 0;
  data.hit_streak = 0;
  data.age = 0;
  data.id = kf_count;
  data.label_id = label_id;
  data.score = score;
  kf_count++;
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
  auto state_mat_xysr = GetXYSRFromTLBR(stateMat);
  kf_.statePost.at<float>(0, 0) = state_mat_xysr[0];
  kf_.statePost.at<float>(1, 0) = state_mat_xysr[1];
  kf_.statePost.at<float>(2, 0) = state_mat_xysr[2];
  kf_.statePost.at<float>(3, 0) = state_mat_xysr[3];
}

KalmanTracker::StateType KalmanTracker::Predict() {
  // predict
  cv::Mat p = kf_.predict();
  data.age += 1;

  if (data.time_since_update > 0) {
    data.hit_streak = 0;
  }
  data.time_since_update += 1;

  StateType predictBox =
      GetTLBRFromXYSR({p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0),
                       p.at<float>(3, 0)});

  history_.push_back(predictBox);
  return history_.back();
}

void KalmanTracker::Update(const StateType& box, int label_id, float score,
                           const cv::Mat& feature) {
  history_.clear();

  data.time_since_update = 0;
  data.hits += 1;
  data.hit_streak += 1;
  data.label_id = label_id;
  data.score = score;
  data.feature = feature.clone();

  // measurement
  StateType state_mat_xysr = GetXYSRFromTLBR(box);
  measurement_.at<float>(0, 0) = state_mat_xysr[0];
  measurement_.at<float>(1, 0) = state_mat_xysr[1];
  measurement_.at<float>(2, 0) = state_mat_xysr[2];
  measurement_.at<float>(3, 0) = state_mat_xysr[3];

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
