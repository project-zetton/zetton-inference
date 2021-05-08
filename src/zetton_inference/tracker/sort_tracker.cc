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
  predictedBoxes.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();) {
    cv::Rect_<float> pBox = (*it).predict();
    if (pBox.x >= 0 && pBox.y >= 0) {
      predictedBoxes.push_back(pBox);
      it++;
    } else {
      it = trackers_.erase(it);
      // cerr << "Box invalid at frame: " << frame_count << endl;
    }
  }

  // associate detections to tracked object (both represented as bounding boxes)
  trkNum = predictedBoxes.size();
  detNum = detections.size();
  iouMatrix.clear();
  iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));
  // compute iou matrix as a distance matrix
  for (unsigned int i = 0; i < trkNum; i++) {
    for (unsigned int j = 0; j < detNum; j++) {
      // use 1-iou because the hungarian algorithm computes a minimum-cost
      // assignment.
      iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detections[j].bbox);
    }
  }

  // solve the assignment problem using hungarian algorithm.
  // the resulting assignment is [track(prediction) : detection], with
  // len=preNum
  tracker::sort::HungarianAlgorithm HungAlgo;
  assignment.clear();
  HungAlgo.Solve(iouMatrix, assignment);

  // find matches, unmatched_detections and unmatched_predictions
  unmatchedTrajectories.clear();
  unmatchedDetections.clear();
  allItems.clear();
  matchedItems.clear();

  if (detNum > trkNum) {
    //	there are unmatched detections
    for (unsigned int n = 0; n < detNum; n++) {
      allItems.insert(n);
    }

    for (unsigned int i = 0; i < trkNum; ++i) {
      matchedItems.insert(assignment[i]);
    }
    set_difference(allItems.begin(), allItems.end(), matchedItems.begin(),
                   matchedItems.end(),
                   std::insert_iterator<std::set<int>>(
                       unmatchedDetections, unmatchedDetections.begin()));
  } else if (detNum < trkNum) {
    // there are unmatched trajectory/predictions
    for (unsigned int i = 0; i < trkNum; ++i) {
      // unassigned label will be set as -1 in the assignment algorithm
      if (assignment[i] == -1) {
        unmatchedTrajectories.insert(i);
      }
    }
  }

  // filter out matched with low IOU
  matchedPairs.clear();
  for (unsigned int i = 0; i < trkNum; ++i) {
    if (assignment[i] == -1)  // pass over invalid values
      continue;
    if (1 - iouMatrix[i][assignment[i]] < iouThreshold_) {
      unmatchedTrajectories.insert(i);
      unmatchedDetections.insert(assignment[i]);
    } else
      matchedPairs.push_back(cv::Point(i, assignment[i]));
  }

  // updating trackers

  // update matched trackers with assigned detections.
  // each prediction is corresponding to a tracker
  int detIdx, trkIdx;
  for (unsigned int i = 0; i < matchedPairs.size(); i++) {
    trkIdx = matchedPairs[i].x;
    detIdx = matchedPairs[i].y;
    trackers_[trkIdx].update(detections[detIdx].bbox);
  }

  // create and initialise new trackers for unmatched detections
  for (auto umd : unmatchedDetections) {
    auto tracker = tracker::sort::KalmanTracker(detections[umd].bbox);
    trackers_.push_back(tracker);
  }

  // get trackers' output
  frameTrackingResult.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();) {
    if (((*it).m_time_since_update < active_age) &&
        ((*it).m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
      TrackingBox res;
      res.box = (*it).get_state();
      res.id = (*it).m_id + 1;
      res.frame = frame_count_;
      frameTrackingResult.push_back(res);
      it++;
    } else
      it++;

    // remove dead tracklet
    if (it != trackers_.end() && (*it).m_time_since_update > max_age_)
      it = trackers_.erase(it);
  }

  return true;
}

}  // namespace inference
}  // namespace zetton
