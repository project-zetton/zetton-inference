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

const KalmanFilter::DetectBox& STrack::GetRect() const { return rect_; }

const STrackState& STrack::GetSTrackState() const { return state_; }

const bool& STrack::IsActivated() const { return is_activated_; }
const float& STrack::GetScore() const { return score_; }

const size_t& STrack::GetTrackId() const { return track_id_; }

const size_t& STrack::GetFrameId() const { return frame_id_; }

const size_t& STrack::GetStartFrameId() const { return start_frame_id_; }

const size_t& STrack::GetTrackletLength() const { return tracklet_len_; }

void STrack::Activate(const size_t& frame_id, const size_t& track_id) {
  kalman_filter_.Init(mean_, covariance_, rect_);

  UpdateRect();

  state_ = STrackState::Tracked;
  if (frame_id == 1) {
    is_activated_ = true;
  }
  track_id_ = track_id;
  frame_id_ = frame_id;
  start_frame_id_ = frame_id;
  tracklet_len_ = 0;
}

void STrack::Reactivate(const STrack& new_track, const size_t& frame_id,
                        const int& new_track_id) {
  kalman_filter_.Update(mean_, covariance_, new_track.GetRect());

  UpdateRect();

  state_ = STrackState::Tracked;
  is_activated_ = true;
  score_ = new_track.GetScore();
  if (0 <= new_track_id) {
    track_id_ = new_track_id;
  }
  frame_id_ = frame_id;
  tracklet_len_ = 0;
}

void STrack::Predict() {
  if (state_ != STrackState::Tracked) {
    mean_[7] = 0;
  }
  kalman_filter_.Predict(mean_, covariance_);
}

void STrack::Update(const STrack& new_track, const size_t& frame_id) {
  kalman_filter_.Update(mean_, covariance_, new_track.GetRect());

  UpdateRect();

  state_ = STrackState::Tracked;
  is_activated_ = true;
  score_ = new_track.GetScore();
  frame_id_ = frame_id;
  tracklet_len_++;
}

void STrack::MarkAsLost() { state_ = STrackState::Lost; }

void STrack::MarkAsRemoved() { state_ = STrackState::Removed; }

void STrack::UpdateRect() {
  float width = mean_[2] * mean_[3];
  float height = mean_[3];
  rect_[0] = mean_[0] - width / 2;
  rect_[1] = mean_[1] - height / 2;
  rect_[2] = mean_[0] + width / 2;
  rect_[3] = mean_[1] + height / 2;
}

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
