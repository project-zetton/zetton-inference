#include "zetton_inference/vision/tracking/bytetrack/byte_tracker.h"

#include <zetton_common/log/log.h>

#include <map>

#include "zetton_inference/vision/tracking/bytetrack/kalman_filter.h"
#include "zetton_inference/vision/util/geometry.h"

namespace zetton {
namespace inference {
namespace vision {

ByteTracker::ByteTracker(const int &frame_rate, const int &track_buffer,
                         const float &track_thresh, const float &high_thresh,
                         const float &match_thresh)
    : frame_id_(0), track_id_count_(0) {
  params_.track_thresh = track_thresh;
  params_.high_thresh = high_thresh;
  params_.match_thresh = match_thresh;
  params_.max_time_lost = static_cast<size_t>(frame_rate / 30.0 * track_buffer);
}

bool ByteTracker::Init(const ByteTrackerParams &params) {
  params_ = params;
  return true;
}

bool ByteTracker::Update(const DetectionResult &detections,
                         TrackingResult &tracks) {
  // do inner update
  auto ret = InnerUpdate(detections);

  // get stracks from inner trackers
  auto stracks = GetSTracks();

  // assign stracks to tracking results
  tracks.Clear();
  tracks.Reserve(static_cast<int>(stracks.size()));
  for (const auto &strack : stracks) {
    tracks.tracking_ids.push_back(strack->GetTrackId());
    tracks.boxes.push_back(GetTLBRFromTLWH(strack->GetRect()));
    tracks.label_ids.push_back(strack->GetLabelId());
    tracks.scores.push_back(strack->GetScore());
  }

  return ret;
}

std::vector<ByteTracker::STrackPtr> ByteTracker::Update(
    const DetectionResult &objects) {
  InnerUpdate(objects);
  return GetSTracks();
}

bool ByteTracker::InnerUpdate(const DetectionResult &objects) {
  // Step 1: Get detections
  frame_id_++;

  // Create new STracks using the result of object detection
  std::vector<STrackPtr> det_stracks;
  std::vector<STrackPtr> det_low_stracks;

  for (std::size_t i = 0; i < objects.boxes.size(); ++i) {
    const auto strack = std::make_shared<bytetrack::STrack>(
        GetTLWHFromTLBR(objects.boxes[i]), objects.label_ids[i],
        objects.scores[i]);
    if (objects.scores[i] >= params_.track_thresh) {
      det_stracks.push_back(strack);
    } else {
      det_low_stracks.push_back(strack);
    }
  }

  // Create lists of existing STrack
  std::vector<STrackPtr> active_stracks;
  std::vector<STrackPtr> non_active_stracks;
  std::vector<STrackPtr> strack_pool;

  for (const auto &tracked_strack : tracked_stracks_) {
    if (!tracked_strack->IsActivated()) {
      non_active_stracks.push_back(tracked_strack);
    } else {
      active_stracks.push_back(tracked_strack);
    }
  }

  strack_pool = JoinSTracks(active_stracks, lost_stracks_);

  // Predict current pose by KF
  for (auto &strack : strack_pool) {
    strack->Predict();
  }

  // Step 2: First association, with IoU
  std::vector<STrackPtr> current_tracked_stracks;
  std::vector<STrackPtr> remain_tracked_stracks;
  std::vector<STrackPtr> remain_det_stracks;
  std::vector<STrackPtr> refind_stracks;

  {
    std::vector<std::vector<int>> matches_idx;
    std::vector<int> unmatch_detection_idx, unmatch_track_idx;

    const auto dists = CalcIoUDistance(strack_pool, det_stracks);
    LinearAssignment(dists, strack_pool.size(), det_stracks.size(),
                     params_.match_thresh, matches_idx, unmatch_track_idx,
                     unmatch_detection_idx);

    for (const auto &match_idx : matches_idx) {
      const auto track = strack_pool[match_idx[0]];
      const auto det = det_stracks[match_idx[1]];
      if (track->GetSTrackState() == bytetrack::STrackState::Tracked) {
        track->Update(*det, frame_id_);
        current_tracked_stracks.push_back(track);
      } else {
        track->Reactivate(*det, frame_id_);
        refind_stracks.push_back(track);
      }
    }

    for (const auto &unmatch_idx : unmatch_detection_idx) {
      remain_det_stracks.push_back(det_stracks[unmatch_idx]);
    }

    for (const auto &unmatch_idx : unmatch_track_idx) {
      if (strack_pool[unmatch_idx]->GetSTrackState() ==
          bytetrack::STrackState::Tracked) {
        remain_tracked_stracks.push_back(strack_pool[unmatch_idx]);
      }
    }
  }

  // Step 3: Second association, using low score dets
  std::vector<STrackPtr> current_lost_stracks;

  {
    std::vector<std::vector<int>> matches_idx;
    std::vector<int> unmatch_track_idx, unmatch_detection_idx;

    const auto dists = CalcIoUDistance(remain_tracked_stracks, det_low_stracks);
    LinearAssignment(dists, remain_tracked_stracks.size(),
                     det_low_stracks.size(), 0.5, matches_idx,
                     unmatch_track_idx, unmatch_detection_idx);

    for (const auto &match_idx : matches_idx) {
      const auto track = remain_tracked_stracks[match_idx[0]];
      const auto det = det_low_stracks[match_idx[1]];
      if (track->GetSTrackState() == bytetrack::STrackState::Tracked) {
        track->Update(*det, frame_id_);
        current_tracked_stracks.push_back(track);
      } else {
        track->Reactivate(*det, frame_id_);
        refind_stracks.push_back(track);
      }
    }

    for (const auto &unmatch_track : unmatch_track_idx) {
      const auto track = remain_tracked_stracks[unmatch_track];
      if (track->GetSTrackState() != bytetrack::STrackState::Lost) {
        track->MarkAsLost();
        current_lost_stracks.push_back(track);
      }
    }
  }

  // Step 4: Init new stracks
  std::vector<STrackPtr> current_removed_stracks;

  {
    std::vector<int> unmatch_detection_idx;
    std::vector<int> unmatch_unconfirmed_idx;
    std::vector<std::vector<int>> matches_idx;

    // Deal with unconfirmed tracks, usually tracks with only one beginning
    // frame
    const auto dists = CalcIoUDistance(non_active_stracks, remain_det_stracks);
    LinearAssignment(dists, non_active_stracks.size(),
                     remain_det_stracks.size(), 0.7, matches_idx,
                     unmatch_unconfirmed_idx, unmatch_detection_idx);

    for (const auto &match_idx : matches_idx) {
      non_active_stracks[match_idx[0]]->Update(
          *remain_det_stracks[match_idx[1]], frame_id_);
      current_tracked_stracks.push_back(non_active_stracks[match_idx[0]]);
    }

    for (const auto &unmatch_idx : unmatch_unconfirmed_idx) {
      const auto track = non_active_stracks[unmatch_idx];
      track->MarkAsRemoved();
      current_removed_stracks.push_back(track);
    }

    // Add new stracks
    for (const auto &unmatch_idx : unmatch_detection_idx) {
      const auto track = remain_det_stracks[unmatch_idx];
      if (track->GetScore() < params_.high_thresh) {
        continue;
      }
      track_id_count_++;
      track->Activate(frame_id_, track_id_count_);
      current_tracked_stracks.push_back(track);
    }
  }

  // Step 5: Update state
  for (const auto &lost_strack : lost_stracks_) {
    if (frame_id_ - lost_strack->GetFrameId() > params_.max_time_lost) {
      lost_strack->MarkAsRemoved();
      current_removed_stracks.push_back(lost_strack);
    }
  }

  tracked_stracks_ = JoinSTracks(current_tracked_stracks, refind_stracks);
  lost_stracks_ =
      SubSTracks(JoinSTracks(SubSTracks(lost_stracks_, tracked_stracks_),
                             current_lost_stracks),
                 removed_stracks_);
  removed_stracks_ = JoinSTracks(removed_stracks_, current_removed_stracks);

  std::vector<STrackPtr> tracked_stracks_out, lost_stracks_out;
  RemoveDuplicateSTracks(tracked_stracks_, lost_stracks_, tracked_stracks_out,
                         lost_stracks_out);
  tracked_stracks_ = tracked_stracks_out;
  lost_stracks_ = lost_stracks_out;

  return true;
}

bool ByteTracker::Update(const DetectionResult &detections,
                         const ReIDResult &features, TrackingResult &tracks) {
  AWARN_F("{} doesn't use ReID result.", Name());
  return Update(detections, tracks);
}

std::string ByteTracker::Name() { return "ByteTracker"; }

std::vector<ByteTracker::STrackPtr> ByteTracker::GetSTracks() {
  std::vector<STrackPtr> output_stracks;
  for (const auto &track : tracked_stracks_) {
    if (track->IsActivated()) {
      output_stracks.push_back(track);
    }
  }

  return output_stracks;
}

ByteTrackerParams *ByteTracker::GetParams() { return &params_; }

std::vector<ByteTracker::STrackPtr> ByteTracker::JoinSTracks(
    const std::vector<STrackPtr> &a_tlist,
    const std::vector<STrackPtr> &b_tlist) const {
  std::map<int, int> exists;
  std::vector<STrackPtr> res;
  for (const auto &elem : a_tlist) {
    exists.emplace(elem->GetTrackId(), 1);
    res.push_back(elem);
  }
  for (const auto &elem : b_tlist) {
    const int &tid = elem->GetTrackId();
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(elem);
    }
  }
  return res;
}

std::vector<ByteTracker::STrackPtr> ByteTracker::SubSTracks(
    const std::vector<STrackPtr> &a_tlist,
    const std::vector<STrackPtr> &b_tlist) const {
  std::map<int, STrackPtr> stracks;
  for (size_t i = 0; i < a_tlist.size(); i++) {
    stracks.emplace(a_tlist[i]->GetTrackId(), a_tlist[i]);
  }

  for (size_t i = 0; i < b_tlist.size(); i++) {
    const int &tid = b_tlist[i]->GetTrackId();
    if (stracks.count(tid) != 0) {
      stracks.erase(tid);
    }
  }

  std::vector<STrackPtr> res;
  std::map<int, STrackPtr>::iterator it;
  for (it = stracks.begin(); it != stracks.end(); ++it) {
    res.push_back(it->second);
  }

  return res;
}

void ByteTracker::RemoveDuplicateSTracks(
    const std::vector<STrackPtr> &a_stracks,
    const std::vector<STrackPtr> &b_stracks, std::vector<STrackPtr> &a_res,
    std::vector<STrackPtr> &b_res) const {
  const auto ious = CalcIoUDistance(a_stracks, b_stracks);

  std::vector<std::pair<size_t, size_t>> overlapping_combinations;
  for (size_t i = 0; i < ious.size(); i++) {
    for (size_t j = 0; j < ious[i].size(); j++) {
      if (ious[i][j] < 0.15) {
        overlapping_combinations.emplace_back(i, j);
      }
    }
  }

  std::vector<bool> a_overlapping(a_stracks.size(), false),
      b_overlapping(b_stracks.size(), false);
  for (const auto &[a_idx, b_idx] : overlapping_combinations) {
    const int timep =
        a_stracks[a_idx]->GetFrameId() - a_stracks[a_idx]->GetStartFrameId();
    const int timeq =
        b_stracks[b_idx]->GetFrameId() - b_stracks[b_idx]->GetStartFrameId();
    if (timep > timeq) {
      b_overlapping[b_idx] = true;
    } else {
      a_overlapping[a_idx] = true;
    }
  }

  for (size_t ai = 0; ai < a_stracks.size(); ai++) {
    if (!a_overlapping[ai]) {
      a_res.push_back(a_stracks[ai]);
    }
  }

  for (size_t bi = 0; bi < b_stracks.size(); bi++) {
    if (!b_overlapping[bi]) {
      b_res.push_back(b_stracks[bi]);
    }
  }
}

void ByteTracker::LinearAssignment(
    const std::vector<std::vector<float>> &cost_matrix,
    const int &cost_matrix_size, const int &cost_matrix_size_size,
    const float &thresh, std::vector<std::vector<int>> &matches,
    std::vector<int> &a_unmatched, std::vector<int> &b_unmatched) const {
  if (cost_matrix.size() == 0) {
    for (int i = 0; i < cost_matrix_size; i++) {
      a_unmatched.push_back(i);
    }
    for (int i = 0; i < cost_matrix_size_size; i++) {
      b_unmatched.push_back(i);
    }
    return;
  }

  std::vector<int> rowsol;
  std::vector<int> colsol;
  execLapjv(cost_matrix, rowsol, colsol, true, thresh);
  for (size_t i = 0; i < rowsol.size(); i++) {
    if (rowsol[i] >= 0) {
      std::vector<int> match;
      match.push_back(i);
      match.push_back(rowsol[i]);
      matches.push_back(match);
    } else {
      a_unmatched.push_back(i);
    }
  }

  for (size_t i = 0; i < colsol.size(); i++) {
    if (colsol[i] < 0) {
      b_unmatched.push_back(i);
    }
  }
}

float ByteTracker::CalcIoU(
    const bytetrack::KalmanFilter::DetectBox &a_tlwh,
    const bytetrack::KalmanFilter::DetectBox &b_tlwh) const {
  // auto a_tlwh = GetTLWHFromTLBR(a_rect);
  // auto b_tlwh = GetTLWHFromTLBR(b_rect);

  const float box_area = (b_tlwh[2] + 1) * (b_tlwh[3] + 1);
  const float iw = std::min(a_tlwh[0] + a_tlwh[2], b_tlwh[0] + b_tlwh[2]) -
                   std::max(a_tlwh[0], b_tlwh[0]) + 1;
  float iou = 0;
  if (iw > 0) {
    const float ih = std::min(a_tlwh[1] + a_tlwh[3], b_tlwh[1] + b_tlwh[3]) -
                     std::max(a_tlwh[1], b_tlwh[1]) + 1;
    if (ih > 0) {
      const float ua = (a_tlwh[0] + a_tlwh[2] - a_tlwh[0] + 1) *
                           (a_tlwh[1] + a_tlwh[3] - a_tlwh[1] + 1) +
                       box_area - iw * ih;
      iou = iw * ih / ua;
    }
  }
  return iou;
}

std::vector<std::vector<float>> ByteTracker::CalcIoUs(
    const std::vector<std::array<float, 4>> &a_tlwhs,
    const std::vector<std::array<float, 4>> &b_tlwhs) const {
  std::vector<std::vector<float>> ious;
  if (a_tlwhs.size() * b_tlwhs.size() == 0) {
    return ious;
  }

  ious.resize(a_tlwhs.size());
  for (auto &iou : ious) {
    iou.resize(b_tlwhs.size());
  }

  for (size_t bi = 0; bi < b_tlwhs.size(); bi++) {
    for (size_t ai = 0; ai < a_tlwhs.size(); ai++) {
      ious[ai][bi] = CalcIoU(b_tlwhs[bi], a_tlwhs[ai]);
    }
  }
  return ious;
}

std::vector<std::vector<float>> ByteTracker::CalcIoUDistance(
    const std::vector<STrackPtr> &a_tracks,
    const std::vector<STrackPtr> &b_tracks) const {
  std::vector<std::array<float, 4>> a_rects, b_rects;
  for (const auto &a_track : a_tracks) {
    a_rects.push_back(a_track->GetRect());
  }

  for (const auto &b_track : b_tracks) {
    b_rects.push_back(b_track->GetRect());
  }

  const auto ious = CalcIoUs(a_rects, b_rects);

  std::vector<std::vector<float>> cost_matrix;
  for (const auto &i : ious) {
    std::vector<float> iou;
    iou.reserve(i.size());
    for (float j : i) {
      iou.push_back(1 - j);
    }
    cost_matrix.push_back(iou);
  }

  return cost_matrix;
}

double ByteTracker::execLapjv(const std::vector<std::vector<float>> &cost,
                              std::vector<int> &rowsol,
                              std::vector<int> &colsol, bool extend_cost,
                              float cost_limit, bool return_cost) const {
  std::vector<std::vector<float>> cost_c;
  cost_c.assign(cost.begin(), cost.end());

  std::vector<std::vector<float>> cost_c_extended;

  int n_rows = cost.size();
  int n_cols = cost[0].size();
  rowsol.resize(n_rows);
  colsol.resize(n_cols);

  int n = 0;
  if (n_rows == n_cols) {
    n = n_rows;
  } else {
    if (!extend_cost) {
      throw std::runtime_error("The `extend_cost` variable should set True");
    }
  }

  if (extend_cost || cost_limit < std::numeric_limits<float>::max()) {
    n = n_rows + n_cols;
    cost_c_extended.resize(n);
    for (size_t i = 0; i < cost_c_extended.size(); i++)
      cost_c_extended[i].resize(n);

    if (cost_limit < std::numeric_limits<float>::max()) {
      for (size_t i = 0; i < cost_c_extended.size(); i++) {
        for (size_t j = 0; j < cost_c_extended[i].size(); j++) {
          cost_c_extended[i][j] = cost_limit / 2.0;
        }
      }
    } else {
      float cost_max = -1;
      for (size_t i = 0; i < cost_c.size(); i++) {
        for (size_t j = 0; j < cost_c[i].size(); j++) {
          if (cost_c[i][j] > cost_max) cost_max = cost_c[i][j];
        }
      }
      for (size_t i = 0; i < cost_c_extended.size(); i++) {
        for (size_t j = 0; j < cost_c_extended[i].size(); j++) {
          cost_c_extended[i][j] = cost_max + 1;
        }
      }
    }

    for (size_t i = n_rows; i < cost_c_extended.size(); i++) {
      for (size_t j = n_cols; j < cost_c_extended[i].size(); j++) {
        cost_c_extended[i][j] = 0;
      }
    }
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        cost_c_extended[i][j] = cost_c[i][j];
      }
    }

    cost_c.clear();
    cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
  }

  double **cost_ptr;
  cost_ptr = new double *[sizeof(double *) * n];
  for (int i = 0; i < n; i++) cost_ptr[i] = new double[sizeof(double) * n];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cost_ptr[i][j] = cost_c[i][j];
    }
  }

  int *x_c = new int[sizeof(int) * n];
  int *y_c = new int[sizeof(int) * n];

  int ret = bytetrack::lapjv_internal(n, cost_ptr, x_c, y_c);
  if (ret != 0) {
    throw std::runtime_error("The result of lapjv_internal() is invalid.");
  }

  double opt = 0.0;

  if (n != n_rows) {
    for (int i = 0; i < n; i++) {
      if (x_c[i] >= n_cols) x_c[i] = -1;
      if (y_c[i] >= n_rows) y_c[i] = -1;
    }
    for (int i = 0; i < n_rows; i++) {
      rowsol[i] = x_c[i];
    }
    for (int i = 0; i < n_cols; i++) {
      colsol[i] = y_c[i];
    }

    if (return_cost) {
      for (size_t i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] != -1) {
          opt += cost_ptr[i][rowsol[i]];
        }
      }
    }
  } else if (return_cost) {
    for (size_t i = 0; i < rowsol.size(); i++) {
      opt += cost_ptr[i][rowsol[i]];
    }
  }

  for (int i = 0; i < n; i++) {
    delete[] cost_ptr[i];
  }
  delete[] cost_ptr;
  delete[] x_c;
  delete[] y_c;

  return opt;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
