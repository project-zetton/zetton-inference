#include "zetton_inference/tracker/sort_tracker.h"

namespace zetton {
namespace inference {

bool SortTracker::Init() {
  tracker::sort::KalmanTracker::kf_count = 0;
  return true;
};

bool SortTracker::Track(const cv::Mat &frame, const ros::Time &timestamp,
                        const ObjectDetectionResults &detections) {
  ++frame_count_;

  if (trackers_.empty()) {
    // initialize kalman trackers using first detections.
    for (unsigned int i = 0; i < detections.size(); i++) {
      auto trk = tracker::sort::KalmanTracker(detections[i].bbox);
      trackers_.push_back(trk);
    }
    return true;
  }

  // get predicted locations from existing trackers.
  predicted_boxes.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();) {
    cv::Rect_<float> pBox = (*it).predict();
    if (pBox.x >= 0 && pBox.y >= 0) {
      predicted_boxes.push_back(pBox);
      it++;
    } else {
      it = trackers_.erase(it);
      // cerr << "Box invalid at frame: " << frame_count << endl;
    }
  }

  // associate detections to tracked object (both represented as bounding boxes)
  num_tracks = predicted_boxes.size();
  num_detections = detections.size();
  iou_matrix.clear();
  iou_matrix.resize(num_tracks, std::vector<double>(num_detections, 0));
  // compute iou matrix as a distance matrix
  for (unsigned int i = 0; i < num_tracks; i++) {
    for (unsigned int j = 0; j < num_detections; j++) {
      // use 1-iou because the hungarian algorithm computes a minimum-cost
      // assignment.
      iou_matrix[i][j] = 1 - GetIOU(predicted_boxes[i], detections[j].bbox);
    }
  }

  // solve the assignment problem using hungarian algorithm.
  // the resulting assignment is [track(prediction) : detection], with
  // len=preNum
  tracker::sort::HungarianAlgorithm HungAlgo;
  assignments.clear();
  HungAlgo.Solve(iou_matrix, assignments);

  // find matches, unmatched_detections and unmatched_predictions
  unmatched_tracks.clear();
  unmatched_detections.clear();
  all_items.clear();
  matched_items.clear();

  if (num_detections > num_tracks) {
    //	there are unmatched detections
    for (unsigned int n = 0; n < num_detections; n++) {
      all_items.insert(n);
    }

    for (unsigned int i = 0; i < num_tracks; ++i) {
      matched_items.insert(assignments[i]);
    }
    set_difference(all_items.begin(), all_items.end(), matched_items.begin(),
                   matched_items.end(),
                   std::insert_iterator<std::set<int>>(
                       unmatched_detections, unmatched_detections.begin()));
  } else if (num_detections < num_tracks) {
    // there are unmatched trajectory/predictions
    for (unsigned int i = 0; i < num_tracks; ++i) {
      // unassigned label will be set as -1 in the assignment algorithm
      if (assignments[i] == -1) {
        unmatched_tracks.insert(i);
      }
    }
  }

  // filter out matched with low IOU
  matched_pairs.clear();
  for (unsigned int i = 0; i < num_tracks; ++i) {
    if (assignments[i] == -1)  // pass over invalid values
      continue;
    if (1 - iou_matrix[i][assignments[i]] < iou_threshold_) {
      unmatched_tracks.insert(i);
      unmatched_detections.insert(assignments[i]);
    } else
      matched_pairs.push_back(cv::Point(i, assignments[i]));
  }

  // update matched trackers with assigned detections.
  // each prediction is corresponding to a tracker
  int detIdx, trkIdx;
  updated_tracks.clear();
  for (unsigned int i = 0; i < matched_pairs.size(); i++) {
    trkIdx = matched_pairs[i].x;
    detIdx = matched_pairs[i].y;
    trackers_[trkIdx].update(detections[detIdx].bbox);
  }

  // create and initialise new trackers for unmatched detections
  for (auto umd : unmatched_detections) {
    auto tracker = tracker::sort::KalmanTracker(detections[umd].bbox);
    trackers_.push_back(tracker);
  }

  // collect visible tracks
  tracking_results.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();) {
    if (((*it).m_time_since_update <= active_age) &&
        ((*it).m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
      TrackingBox res;
      res.box = (*it).get_state();
      res.id = (*it).m_id + 1;
      res.frame = frame_count_;
      tracking_results.push_back(res);
      ++it;
    } else {
      ++it;
    }

    // remove dead tracklet
    if (it != trackers_.end() && (*it).m_time_since_update > max_age_) {
      it = trackers_.erase(it);
    }
  }

  return true;
}

}  // namespace inference
}  // namespace zetton
