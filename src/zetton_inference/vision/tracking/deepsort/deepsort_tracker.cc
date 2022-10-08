#include "zetton_inference/vision/tracking/deepsort/deepsort_tracker.h"

#include <cstddef>

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/tracking/deepsort/hungarian.h"

namespace zetton {
namespace inference {
namespace vision {

bool DeepSORTTracker::Update(const DetectionResult &detections,
                             TrackingResult &tracks) {
  AERROR_F("{} needs ReID result as input", Name());
  return false;
}

bool DeepSORTTracker::Update(const DetectionResult &detections,
                             const ReIDResult &features,
                             TrackingResult &tracking_results) {
  // construct tracking results by detection results
  tracking_results.Clear();
  tracking_results.Reserve(static_cast<int>(detections.boxes.size()));
  int index = 0;
  for (std::size_t i = 0; i < detections.boxes.size(); ++i) {
    tracking_results.boxes[i] = detections.boxes[i];
    tracking_results.scores[i] = detections.scores[i];
    tracking_results.label_ids[i] = detections.label_ids[i];
    tracking_results.features[i] = features.features[i];
    index++;
  }
  // construct kalman tracks by tracking results if no previous tracks
  if (kalman_boxes_.empty()) {
    for (std::size_t i = 0; i < tracking_results.boxes.size(); ++i) {
      auto tracker = deepsort::KalmanTracker(tracking_results.boxes[i],
                                             tracking_results.label_ids[i],
                                             tracking_results.scores[i]);
      tracker.data.feature = tracking_results.features[i].clone();
      tracking_results.tracking_ids[i] = tracker.data.id;
      kalman_boxes_.push_back(tracker);
    }
    return false;
  }

  // check predicted tracks from kalman filter
  TrackingResult predict_boxes;
  for (auto it = kalman_boxes_.begin(); it != kalman_boxes_.end();) {
    auto pBox = (*it).Predict();

    // check if the predicted track is NaN
    bool is_nan = (std::isnan(pBox[0])) || (std::isnan(pBox[1])) ||
                  (std::isnan(pBox[2])) || (std::isnan(pBox[3]));
    // check if the predicted track is out of image
    bool is_bound = (pBox[0] > static_cast<float>(params_.size[0])) ||
                    (pBox[1] > static_cast<float>(params_.size[1])) ||
                    (pBox[2] < 0) || (pBox[3] < 0);
    // check if the width or height of predicted track is invalid
    bool is_illegal = (pBox[2] < pBox[0]) || (pBox[3] < pBox[1]);
    // check if the predicted track is too oldjlk;w
    bool time_since_update = it->data.time_since_update > params_.max_age;

    if (!(time_since_update || is_nan || is_bound || is_illegal)) {
      // accept the predicted track if it is valid
      predict_boxes.boxes.push_back(pBox);
      predict_boxes.label_ids.push_back(it->data.label_id);
      predict_boxes.scores.push_back(it->data.score);
      predict_boxes.tracking_ids.push_back(it->data.id);
      predict_boxes.features.push_back(it->data.feature);
      it++;
    } else {
      // remove the predicted track if it is invalid
      it = kalman_boxes_.erase(it);
    }
  }

  // associate detection results with predicted tracks
  std::set<int> unassigned_detections;
  std::set<int> unassigned_tracks;
  std::vector<cv::Point> assigned_pairs;
  FeatureMatching(tracking_results, predict_boxes, unassigned_detections,
                  unassigned_tracks, assigned_pairs);
  IOUMatching(tracking_results, predict_boxes, unassigned_detections,
              unassigned_tracks, assigned_pairs);

  // update tracks with assigned detection results
  for (auto &assigned_pair : assigned_pairs) {
    int trk_id = assigned_pair.x;
    int det_id = assigned_pair.y;
    kalman_boxes_[trk_id].Update(
        tracking_results.boxes[det_id], tracking_results.label_ids[det_id],
        tracking_results.scores[det_id], tracking_results.features[det_id]);
    tracking_results.tracking_ids[det_id] = kalman_boxes_[trk_id].data.id;
  }

  // create new tracks for unassigned detection results
  for (const auto &umd : unassigned_detections) {
    auto tracker = deepsort::KalmanTracker(tracking_results.boxes[umd],
                                           tracking_results.label_ids[umd],
                                           tracking_results.scores[umd]);
    tracking_results.tracking_ids[umd] = tracker.data.id;
    tracker.data.feature = tracking_results.features[umd].clone();
    kalman_boxes_.push_back(tracker);
  }

  return true;
}

std::string DeepSORTTracker::Name() { return "DeepSORTTracker"; }

DeepSORTTrackerParams *DeepSORTTracker::GetParams() { return &params_; }

float DeepSORTTracker::IOUCalculate(
    const deepsort::KalmanTracker::StateType &det_a,
    const deepsort::KalmanTracker::StateType &det_b) {
  cv::Point2f center_a((det_a[0] + det_a[2]) / 2, (det_a[1] + det_b[3]) / 2);
  cv::Point2f center_b((det_b[0] + det_b[2]) / 2, (det_b[1] + det_b[3]) / 2);
  cv::Point2f left_up(std::min(det_a[0], det_b[0]),
                      std::min(det_a[1], det_b[1]));
  cv::Point2f right_down(std::max(det_a[2], det_b[2]),
                         std::max(det_a[3], det_b[3]));
  float distance_d = (center_a - center_b).x * (center_a - center_b).x +
                     (center_a - center_b).y * (center_a - center_b).y;
  float distance_c = (left_up - right_down).x * (left_up - right_down).x +
                     (left_up - right_down).y * (left_up - right_down).y;
  float inter_l = det_a[0] > det_b[0] ? det_a[0] : det_b[0];
  float inter_t = det_a[1] > det_b[1] ? det_a[1] : det_b[1];
  float inter_r = det_a[2] < det_b[2] ? det_a[2] : det_b[2];
  float inter_b = det_a[3] < det_b[3] ? det_a[3] : det_b[3];
  if (inter_b < inter_t || inter_r < inter_l) return 0;
  float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
  float union_area = (det_a[2] - det_a[0]) * (det_a[3] - det_a[1]) +
                     (det_b[2] - det_b[0]) * (det_b[3] - det_b[1]) - inter_area;
  if (union_area == 0)
    return 0;
  else
    return inter_area / union_area - distance_d / distance_c;
}

void DeepSORTTracker::Alignment(std::vector<std::vector<double>> cost_matrix,
                                std::set<int> &unassigned_detections,
                                std::set<int> &unassigned_tracks,
                                std::vector<cv::Point> &assigned_pairs,
                                int det_num, int trk_num, bool b_iou) {
  std::vector<int> assignment;
  deepsort::HungarianAlgorithm hungarian;
  hungarian.Solve(cost_matrix, assignment);

  std::set<int> allItems;
  std::set<int> matchedItems;

  if (b_iou) {
    std::vector<int> detection_index(unassigned_detections.size());
    std::vector<int> tracker_index(unassigned_tracks.size());
    int idx = 0;
    for (const int &umd : unassigned_detections) {
      detection_index[idx] = umd;
      idx++;
    }
    idx = 0;
    for (const int &umt : unassigned_tracks) {
      tracker_index[idx] = umt;
      idx++;
    }
    unassigned_detections.clear();
    unassigned_tracks.clear();
    if (det_num > trk_num) {  //	there are unmatched detections
      for (int n = 0; n < det_num; n++) allItems.insert(detection_index[n]);

      for (int i = 0; i < trk_num; ++i)
        matchedItems.insert(detection_index[assignment[i]]);

      set_difference(allItems.begin(), allItems.end(), matchedItems.begin(),
                     matchedItems.end(),
                     std::insert_iterator<std::set<int>>(
                         unassigned_detections, unassigned_detections.begin()));
    } else if (det_num <
               trk_num) {  // there are unmatched trajectory/predictions
      for (int i = 0; i < trk_num; ++i)
        if (assignment[i] == -1)  // unassigned label will be set as -1 in the
                                  // assignment algorithm
          unassigned_tracks.insert(tracker_index[i]);
    }
    for (int i = 0; i < trk_num; ++i) {
      if (assignment[i] == -1)  // pass over invalid values
        continue;
      if (1 - cost_matrix[i][assignment[i]] < params_.iou_threshold) {
        unassigned_tracks.insert(tracker_index[i]);
        unassigned_detections.insert(detection_index[assignment[i]]);
      } else
        assigned_pairs.emplace_back(
            cv::Point(tracker_index[i], detection_index[assignment[i]]));
    }
  } else {
    if (det_num > trk_num) {  //	there are unmatched detections
      for (int n = 0; n < det_num; n++) allItems.insert(n);

      for (int i = 0; i < trk_num; ++i) matchedItems.insert(assignment[i]);

      set_difference(allItems.begin(), allItems.end(), matchedItems.begin(),
                     matchedItems.end(),
                     std::insert_iterator<std::set<int>>(
                         unassigned_detections, unassigned_detections.begin()));
    } else if (det_num <
               trk_num) {  // there are unmatched trajectory/predictions
      for (int i = 0; i < trk_num; ++i)
        if (assignment[i] == -1)  // unassigned label will be set as -1 in the
                                  // assignment algorithm
          unassigned_tracks.insert(i);
    }
    for (int i = 0; i < trk_num; ++i) {
      if (assignment[i] == -1)  // pass over invalid values
        continue;
      if (1 - cost_matrix[i][assignment[i]] < params_.sim_threshold) {
        unassigned_tracks.insert(i);
        unassigned_detections.insert(assignment[i]);
      } else
        assigned_pairs.emplace_back(cv::Point(i, assignment[i]));
    }
  }
}

void DeepSORTTracker::IOUMatching(const TrackingResult &detected_boxes,
                                  const TrackingResult &predict_boxes,
                                  std::set<int> &unassigned_detections,
                                  std::set<int> &unassigned_tracks,
                                  std::vector<cv::Point> &assigned_pairs) {
  int det_num = static_cast<int>(unassigned_detections.size());
  int trk_num = static_cast<int>(unassigned_tracks.size());
  if (det_num == 0 or trk_num == 0) {
    return;
  }
  std::vector<std::vector<double>> iou_mat(trk_num,
                                           std::vector<double>(det_num, 0));
  // compute iou matrix as a distance matrix
  int i = 0;
  for (const int &umt : unassigned_tracks) {
    int j = 0;
    for (const int &umd : unassigned_detections) {
      if (predict_boxes.label_ids[umt] == detected_boxes.label_ids[umd] ||
          params_.agnostic) {
        iou_mat[i][j] = 1 - IOUCalculate(predict_boxes.boxes[umt],
                                         detected_boxes.boxes[umd]);
      } else
        iou_mat[i][j] = 1;
      j++;
    }
    i++;
  }
  Alignment(iou_mat, unassigned_detections, unassigned_tracks, assigned_pairs,
            det_num, trk_num, true);
}

void DeepSORTTracker::FeatureMatching(const TrackingResult &detected_boxes,
                                      const TrackingResult &predict_boxes,
                                      std::set<int> &unassigned_detections,
                                      std::set<int> &unassigned_tracks,
                                      std::vector<cv::Point> &assigned_pairs) {
  int det_num = static_cast<int>(detected_boxes.boxes.size());
  int trk_num = static_cast<int>(predict_boxes.boxes.size());
  std::vector<std::vector<double>> similar_mat(trk_num,
                                               std::vector<double>(det_num, 0));
  // compute iou matrix as a distance matrix
  for (int i = 0; i < trk_num; i++) {
    for (int j = 0; j < det_num; j++) {
      // use 1-iou because the hungarian algorithm computes a minimum-cost
      // assignment.
      if (predict_boxes.label_ids[i] == detected_boxes.label_ids[j] ||
          params_.agnostic) {
        similar_mat[i][j] =
            1 - predict_boxes.features[i].dot(detected_boxes.features[j]);
      } else {
        similar_mat[i][j] = 1;
      }
    }
  }
  Alignment(similar_mat, unassigned_detections, unassigned_tracks,
            assigned_pairs, det_num, trk_num, false);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
