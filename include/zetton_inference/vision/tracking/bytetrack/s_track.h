#pragma once

#include "zetton_inference/vision/tracking/bytetrack/kalman_filter.h"

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

/// \brief state of a track
enum class STrackState {
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3,
};

/// \brief track of a single object
class STrack {
 public:
  /// \brief constructor
  /// \param rect bounding box of the object in [x,y,w,h] style
  /// \param score detection score
  STrack(const KalmanFilter::DetectBox& rect, const int32_t label_id,
         const float& score);

  /// \brief destructor
  ~STrack() = default;

 public:
  /// \brief get the bounding box of the object in [x,y,w,h] style
  const KalmanFilter::DetectBox& GetRect() const;
  /// \brief get the state of the track
  const STrackState& GetSTrackState() const;

  /// \brief whether the track has been activated
  const bool& IsActivated() const;
  /// \brief get the detection label id of the track
  const int32_t& GetLabelId() const;
  /// \brief get the detection score of the track
  const float& GetScore() const;
  /// \brief get the track id
  const size_t& GetTrackId() const;
  /// \brief get the frame id of the last tracking
  const size_t& GetFrameId() const;
  /// \brief get the frame id of the first tracking
  const size_t& GetStartFrameId() const;
  /// \brief get the number of frames that the track has been tracked
  const size_t& GetTrackletLength() const;

 public:
  /// \brief activate the track with the given frame id and track id
  /// \param frame_id frame id of the activation
  /// \param track_id track id of the activation
  void Activate(const size_t& frame_id, const size_t& track_id);
  /// \brief re-activate the track with the given track
  /// \param new_track track to activate the current track
  /// \param frame_id frame id of the activation
  /// \param new_track_id track id of the new track
  void Reactivate(const STrack& new_track, const size_t& frame_id,
                  const int& new_track_id = -1);

  /// \brief predict the bounding box of the object in the next frame
  void Predict();
  /// \brief update the track with the given detection
  void Update(const STrack& new_track, const size_t& frame_id);

  /// \brief mark the track as lost
  void MarkAsLost();
  /// \brief mark the track as removed
  void MarkAsRemoved();

  /// \brief update rect with the current state of the kalman filter
  void UpdateRect();

 private:
  /// \brief kalman filter implementation
  KalmanFilter kalman_filter_;
  /// \brief filter state mean of the track
  KalmanFilter::StateMean mean_;
  /// \brief filter state covariance of the track
  KalmanFilter::StateCov covariance_;
  /// \brief bounding box of the object in [x,y,w,h] style
  KalmanFilter::DetectBox rect_;
  /// \brief tracking state of the track
  STrackState state_;

  /// \brief whether the track has been activated
  bool is_activated_;
  /// \brief detection label id of the track
  int32_t label_id_;
  /// \brief detection score of the track
  float score_;
  /// \brief track id of the track
  size_t track_id_;
  /// \brief frame id of the last tracking
  size_t frame_id_;
  /// \brief frame id of the first tracking
  size_t start_frame_id_;
  /// \brief number of frames that the track has been tracked
  size_t tracklet_len_;
};

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
