#pragma once

#include "zetton_inference/vision/tracking/bytetrack/kalman_filter.h"

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

enum class STrackState {
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3,
};

class STrack {
 public:
  STrack(const KalmanFilter::DetectBox& rect, const float& score);
  ~STrack() = default;

  const KalmanFilter::DetectBox& getRect() const;
  const STrackState& getSTrackState() const;

  const bool& isActivated() const;
  const float& getScore() const;
  const size_t& getTrackId() const;
  const size_t& getFrameId() const;
  const size_t& getStartFrameId() const;
  const size_t& getTrackletLength() const;

  void activate(const size_t& frame_id, const size_t& track_id);
  void reActivate(const STrack& new_track, const size_t& frame_id,
                  const int& new_track_id = -1);

  void predict();
  void update(const STrack& new_track, const size_t& frame_id);

  void markAsLost();
  void markAsRemoved();

  KalmanFilter::DetectBox GetXYAHFromLTRB(const KalmanFilter::DetectBox& ltrb);

 private:
  KalmanFilter kalman_filter_;
  KalmanFilter::StateMean mean_;
  KalmanFilter::StateCov covariance_;

  KalmanFilter::DetectBox rect_;
  STrackState state_;

  bool is_activated_;
  float score_;
  size_t track_id_;
  size_t frame_id_;
  size_t start_frame_id_;
  size_t tracklet_len_;

  void updateRect();
};

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
