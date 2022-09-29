#include "zetton_inference/vision/tracking/bytetrack/s_track.h"

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

STrack::STrack(const KalmanFilter::DetectBox& rect, const float& score)
    : kalman_filter_(),
      mean_(),
      covariance_(),
      rect_(rect),
      state_(STrackState::New),
      is_activated_(false),
      score_(score),
      track_id_(0),
      frame_id_(0),
      start_frame_id_(0),
      tracklet_len_(0) {}

const KalmanFilter::DetectBox& STrack::getRect() const { return rect_; }

const STrackState& STrack::getSTrackState() const { return state_; }

const bool& STrack::isActivated() const { return is_activated_; }
const float& STrack::getScore() const { return score_; }

const size_t& STrack::getTrackId() const { return track_id_; }

const size_t& STrack::getFrameId() const { return frame_id_; }

const size_t& STrack::getStartFrameId() const { return start_frame_id_; }

const size_t& STrack::getTrackletLength() const { return tracklet_len_; }

void STrack::activate(const size_t& frame_id, const size_t& track_id) {
  kalman_filter_.initiate(mean_, covariance_, rect_);

  updateRect();

  state_ = STrackState::Tracked;
  if (frame_id == 1) {
    is_activated_ = true;
  }
  track_id_ = track_id;
  frame_id_ = frame_id;
  start_frame_id_ = frame_id;
  tracklet_len_ = 0;
}

void STrack::reActivate(const STrack& new_track, const size_t& frame_id,
                        const int& new_track_id) {
  kalman_filter_.update(mean_, covariance_, new_track.getRect());

  updateRect();

  state_ = STrackState::Tracked;
  is_activated_ = true;
  score_ = new_track.getScore();
  if (0 <= new_track_id) {
    track_id_ = new_track_id;
  }
  frame_id_ = frame_id;
  tracklet_len_ = 0;
}

void STrack::predict() {
  if (state_ != STrackState::Tracked) {
    mean_[7] = 0;
  }
  kalman_filter_.predict(mean_, covariance_);
}

void STrack::update(const STrack& new_track, const size_t& frame_id) {
  kalman_filter_.update(mean_, covariance_, new_track.getRect());

  updateRect();

  state_ = STrackState::Tracked;
  is_activated_ = true;
  score_ = new_track.getScore();
  frame_id_ = frame_id;
  tracklet_len_++;
}

void STrack::markAsLost() { state_ = STrackState::Lost; }

void STrack::markAsRemoved() { state_ = STrackState::Removed; }

void STrack::updateRect() {
  float width = mean_[2] * mean_[3];
  float height = mean_[3];
  rect_[0] = mean_[0] - width / 2;
  rect_[1] = mean_[1] - height / 2;
  rect_[2] = mean_[0] + width / 2;
  rect_[3] = mean_[1] + height / 2;
}

KalmanFilter::DetectBox STrack::GetXYAHFromLTRB(
    const KalmanFilter::DetectBox& ltrb) {
  int width = ltrb[2] - ltrb[0];
  int height = ltrb[3] - ltrb[1];
  return {};
}

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
