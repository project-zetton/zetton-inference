#pragma once

#include <memory>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/tracking/bytetrack/kalman_filter.h"
#include "zetton_inference/vision/tracking/bytetrack/lapjv.h"
#include "zetton_inference/vision/tracking/bytetrack/s_track.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief ByteTrack tracker
class ByteTracker {
 public:
  using STrackPtr = std::shared_ptr<bytetrack::STrack>;

  /// \brief constructor
  ByteTracker(const int &frame_rate = 30, const int &track_buffer = 30,
              const float &track_thresh = 0.5, const float &high_thresh = 0.6,
              const float &match_thresh = 0.8);
  /// \brief destructor
  ~ByteTracker();

  /// \brief update the tracker with the given detection results and return the
  /// tracking results
  /// \param objects detection results
  /// \return tracking results with [x,y,w,h] box format
  std::vector<STrackPtr> Update(const DetectionResult &objects);

 private:
  /// \brief merge two vectors of STrackPtr into one (a + b)
  /// \param a_tlist vector a
  /// \param b_tlist vector b
  /// \return merged vector of STrackPtr
  std::vector<STrackPtr> JoinSTracks(
      const std::vector<STrackPtr> &a_tlist,
      const std::vector<STrackPtr> &b_tlist) const;

  /// \brief substraction of two vectors of STrackPtr (a - b)
  /// \param a_tlist vector a
  /// \param b_tlist vector b
  /// \return substracted vector of STrackPtr
  std::vector<STrackPtr> SubSTracks(
      const std::vector<STrackPtr> &a_tlist,
      const std::vector<STrackPtr> &b_tlist) const;

  /// \brief remove duplicated tracks in the given vectors of STrackPtr
  /// \param a_strakcs vector of STrackPtr
  /// \return vectors of STrackPtr without duplicated tracks
  void RemoveDuplicateSTracks(const std::vector<STrackPtr> &a_stracks,
                              const std::vector<STrackPtr> &b_stracks,
                              std::vector<STrackPtr> &a_res,
                              std::vector<STrackPtr> &b_res) const;

  /// \brief associate the given detection results with the given tracks
  /// \param cost_matrix cost matrix of the association
  /// \param cost_matrix_size number of rows of the cost matrix (number of
  /// detections)
  /// \param cost_matrix_size_size number of columns of the cost matrix (number
  /// of tracks)
  /// \param thresh threshold of the association
  /// \param matches matches of the association
  /// \param b_unmatched unmatched tracks during the association
  /// \param a_unmatched unmatched detections during the association
  void LinearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                        const int &cost_matrix_size,
                        const int &cost_matrix_size_size, const float &thresh,
                        std::vector<std::vector<int>> &matches,
                        std::vector<int> &b_unmatched,
                        std::vector<int> &a_unmatched) const;

  /// \brief calculate the cost matrix of the association by IoU
  /// \param a_stracks tracks
  /// \param b_stracks another tracks
  std::vector<std::vector<float>> CalcIoUDistance(
      const std::vector<STrackPtr> &a_tracks,
      const std::vector<STrackPtr> &b_tracks) const;

  /// \brief calculate the IoU of two bounding boxes
  /// \param a_rect bounding box a
  /// \param b_rect bounding box b
  float CalcIoU(const bytetrack::KalmanFilter::DetectBox &a_rect,
                const bytetrack::KalmanFilter::DetectBox &b_rect) const;

  /// \brief calculate the cost matrix of the association by IoU
  /// \param a_stracks tracks
  /// \param b_stracks another tracks
  std::vector<std::vector<float>> CalcIoUs(
      const std::vector<bytetrack::KalmanFilter::DetectBox> &a_rect,
      const std::vector<bytetrack::KalmanFilter::DetectBox> &b_rect) const;

  /// \brief solve the assignment problem by Hungarian algorithm
  double execLapjv(const std::vector<std::vector<float>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol,
                   bool extend_cost = false,
                   float cost_limit = std::numeric_limits<float>::max(),
                   bool return_cost = true) const;

 private:
  /// \brief threshold allowed for tracking
  const float track_thresh_;
  /// \brief threshold for high-confidence measurements
  const float high_thresh_;
  /// \brief threhosld for detection-to-track matching
  const float match_thresh_;
  /// \brief maximum invisible duration for a track to be marked as lost
  const size_t max_time_lost_;

  /// \brief current frame id
  std::size_t frame_id_;
  /// \brief counter of all tracks
  std::size_t track_id_count_;

  /// \brief current alive tracks
  std::vector<STrackPtr> tracked_stracks_;
  /// \brief lost tracks
  std::vector<STrackPtr> lost_stracks_;
  /// \brief removed tracks
  std::vector<STrackPtr> removed_stracks_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
