#pragma once

#include <memory>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/tracking/bytetrack/kalman_filter.h"
#include "zetton_inference/vision/tracking/bytetrack/lapjv.h"
#include "zetton_inference/vision/tracking/bytetrack/s_track.h"

namespace zetton {
namespace inference {
namespace vision {

class BYTETracker {
 public:
  using STrackPtr = std::shared_ptr<bytetrack::STrack>;

  BYTETracker(const int &frame_rate = 30, const int &track_buffer = 30,
              const float &track_thresh = 0.5, const float &high_thresh = 0.6,
              const float &match_thresh = 0.8);
  ~BYTETracker();

  std::vector<STrackPtr> Update(const DetectionResult &objects);

 private:
  std::vector<STrackPtr> JoinSTracks(
      const std::vector<STrackPtr> &a_tlist,
      const std::vector<STrackPtr> &b_tlist) const;

  std::vector<STrackPtr> SubSTracks(
      const std::vector<STrackPtr> &a_tlist,
      const std::vector<STrackPtr> &b_tlist) const;

  void RemoveDuplicateSTracks(const std::vector<STrackPtr> &a_stracks,
                              const std::vector<STrackPtr> &b_stracks,
                              std::vector<STrackPtr> &a_res,
                              std::vector<STrackPtr> &b_res) const;

  void LinearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                        const int &cost_matrix_size,
                        const int &cost_matrix_size_size, const float &thresh,
                        std::vector<std::vector<int>> &matches,
                        std::vector<int> &b_unmatched,
                        std::vector<int> &a_unmatched) const;

  std::vector<std::vector<float>> CalcIoUDistance(
      const std::vector<STrackPtr> &a_tracks,
      const std::vector<STrackPtr> &b_tracks) const;

  float CalcIoU(const bytetrack::KalmanFilter::DetectBox &a_rect,
                const bytetrack::KalmanFilter::DetectBox &b_rect) const;

  std::vector<std::vector<float>> CalcIoUs(
      const std::vector<bytetrack::KalmanFilter::DetectBox> &a_rect,
      const std::vector<bytetrack::KalmanFilter::DetectBox> &b_rect) const;

  double execLapjv(const std::vector<std::vector<float>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol,
                   bool extend_cost = false,
                   float cost_limit = std::numeric_limits<float>::max(),
                   bool return_cost = true) const;

 private:
  const float track_thresh_;
  const float high_thresh_;
  const float match_thresh_;
  const size_t max_time_lost_;

  size_t frame_id_;
  size_t track_id_count_;

  std::vector<STrackPtr> tracked_stracks_;
  std::vector<STrackPtr> lost_stracks_;
  std::vector<STrackPtr> removed_stracks_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
