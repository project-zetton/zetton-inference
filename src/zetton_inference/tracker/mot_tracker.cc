#include "zetton_inference/tracker/mot_tracker.h"

#include <cstddef>
#include <stdexcept>

#include "zetton_common/util/ros_util.h"

namespace zetton {
namespace inference {

bool MotTracker::Init() {
  load_config(&nh_);
  opt_tracker = tracker::OpticalFlow(opt_param);
}

std::vector<tracker::LocalObject> MotTracker::update_bbox_by_tracker(
    const cv::Mat &img, const ros::Time &update_time) {
  // update the tracker and the database of each tracking object
  track_bbox_by_optical_flow(img, update_time, true);

  // remove the tracker that loses track and also check whether enable opt(to
  // avoid degeneration under occlusion)
  std::vector<tracker::LocalObject> dead_tracker = remove_dead_trackers();

  // udpate overlap flag
  // TODO might remove this part
  update_overlap_flag();

  // summary
  report_local_object();

  return dead_tracker;
}

void MotTracker::update_bbox_by_detector(
    const cv::Mat &img, const std::vector<cv::Rect2d> &bboxes,
    const std::vector<float> features_detector, const ros::Time &update_time) {
  AINFO_F("******Update Bbox by Detector******");

  // maximum block size, to perform augumentation and rectification of the
  // tracker block
  cv::Rect2d block_max(0, 0, img.cols, img.rows);

  // update by optical flow first
  track_bbox_by_optical_flow(img, update_time, false);

  // associate the detected bboxes with tracking bboxes
  std::vector<tracker::AssociationVector> all_detected_bbox_ass_vec;
  std::vector<Eigen::VectorXf> feats_eigen =
      tracker::feature_vector_to_eigen(features_detector);

  detector_and_tracker_association(bboxes, block_max, feats_eigen,
                                   all_detected_bbox_ass_vec);

  // local object list management
  manage_local_objects_list_by_reid_detector(
      bboxes, block_max, feats_eigen, features_detector, img, update_time,
      all_detected_bbox_ass_vec);

  // summary
  report_local_object();
}

void MotTracker::load_config(ros::NodeHandle *nh_) {
  GPARAM("/zetton_inference/mot_tracker/tracker/track_fail_timeout_tick",
         track_fail_timeout_tick);
  GPARAM("/zetton_inference/mot_tracker/tracker/bbox_overlap_ratio",
         bbox_overlap_ratio_threshold);
  GPARAM("/zetton_inference/mot_tracker/tracker/detector_update_timeout_tick",
         detector_update_timeout_tick);
  GPARAM("/zetton_inference/mot_tracker/tracker/stop_opt_timeout",
         stop_opt_timeout);
  GPARAM("/zetton_inference/mot_tracker/tracker/detector_bbox_padding",
         detector_bbox_padding);
  GPARAM("/zetton_inference/mot_tracker/tracker/reid_match_threshold",
         reid_match_threshold);
  GPARAM("/zetton_inference/mot_tracker/tracker/reid_match_bbox_dis",
         reid_match_bbox_dis);
  GPARAM("/zetton_inference/mot_tracker/tracker/reid_match_bbox_size_diff",
         reid_match_bbox_size_diff);

  GPARAM("/zetton_inference/mot_tracker/local_database/height_width_ratio_min",
         height_width_ratio_min);
  GPARAM("/zetton_inference/mot_tracker/local_database/height_width_ratio_max",
         height_width_ratio_max);
  GPARAM("/zetton_inference/mot_tracker/local_database/record_interval",
         record_interval);
  GPARAM("/zetton_inference/mot_tracker/local_database/feature_smooth_ratio",
         feature_smooth_ratio);

  // kalman filter
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/q_xy", kf_param.q_xy);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/q_wh", kf_param.q_wh);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/p_xy_pos",
         kf_param.p_xy_pos);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/p_xy_dp",
         kf_param.p_xy_dp);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/p_wh_size",
         kf_param.p_wh_size);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/p_wh_ds",
         kf_param.p_wh_ds);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/r_theta",
         kf_param.r_theta);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/r_f", kf_param.r_f);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/r_tx", kf_param.r_tx);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/r_ty", kf_param.r_ty);
  GPARAM("/zetton_inference/mot_tracker/kalman_filter/residual_threshold",
         kf_param.residual_threshold);  // TODO better usage of residual

  // optical tracker
  GPARAM("/zetton_inference/mot_tracker/optical_flow/min_keypoints_to_track",
         opt_param.min_keypoints_to_track);
  GPARAM("/zetton_inference/mot_tracker/optical_flow/keypoints_num_factor_area",
         opt_param.keypoints_num_factor_area);
  GPARAM("/zetton_inference/mot_tracker/optical_flow/corner_detector_max_num",
         opt_param.corner_detector_max_num);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/"
      "corner_detector_quality_level",
      opt_param.corner_detector_quality_level);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/corner_detector_min_distance",
      opt_param.corner_detector_min_distance);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/corner_detector_block_size",
      opt_param.corner_detector_block_size);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/corner_detector_use_harris",
      opt_param.corner_detector_use_harris);
  GPARAM("/zetton_inference/mot_tracker/optical_flow/corner_detector_k",
         opt_param.corner_detector_k);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/min_keypoints_to_cal_H_mat",
      opt_param.min_keypoints_to_cal_H_mat);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/"
      "min_keypoints_for_motion_estimation",
      opt_param.min_keypoints_for_motion_estimation);
  GPARAM(
      "/zetton_inference/mot_tracker/optical_flow/"
      "min_pixel_dis_square_for_scene_point",
      opt_param.min_pixel_dis_square_for_scene_point);
  GPARAM("/zetton_inference/mot_tracker/optical_flow/use_resize",
         opt_param.use_resize);
  GPARAM("/zetton_inference/mot_tracker/optical_flow/resize_factor",
         opt_param.resize_factor);
}

bool MotTracker::update_local_database(tracker::LocalObject &local_object,
                                       const cv::Mat &img_block) {
  // two criterion to manage local database:
  // 1. appropriate width/height ratio
  // 2. fulfill the minimum time interval
  if (1.0 * img_block.rows / img_block.cols > height_width_ratio_min &&
      1.0 * img_block.rows / img_block.cols < height_width_ratio_max &&
      local_object.database_update_timer.toc() > record_interval) {
    local_object.img_blocks.push_back(img_block);
    local_object.database_update_timer.tic();
    AINFO_F("Adding an image to the datebase id: ", local_object.id);
    return true;
  } else {
    return false;
  }
}

void MotTracker::update_overlap_flag() {
  // lock_guard<mutex> lk(mtx); //lock the thread
  for (auto &lo : local_objects_list) {
    lo.is_overlap = false;
  }

  for (auto &lo : local_objects_list) {
    if (lo.is_overlap) {
      continue;
    }

    for (auto lo2 : local_objects_list) {
      if (lo.id == lo2.id) {
        continue;
      }
      if ((tracker::BboxPadding(lo.bbox, match_centroid_padding) &
           tracker::BboxPadding(lo2.bbox, match_centroid_padding))
              .area() > 1e-3)  // FIXME hard code in here
      {
        lo.is_overlap = true;
        lo2.is_overlap = true;
      }
    }
  }
}

void MotTracker::track_bbox_by_optical_flow(const cv::Mat &img,
                                            const ros::Time &update_time,
                                            bool update_database) {
  cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(img.cols, img.rows));

  // get the bbox measurement by optical flow
  opt_tracker.update(img, local_objects_list);

  // update each tracking object in tracking list by kalman filter
  for (auto &lo : local_objects_list) {
    lo.track_bbox_by_optical_flow(update_time);
    bbox_rect(block_max);
    // std::cout << lo.bbox << std::endl;

    // update database
    if (lo.is_track_succeed && update_database) {
      update_local_database(lo, img(lo.bbox));
    }
  }
}

std::vector<tracker::LocalObject> MotTracker::remove_dead_trackers() {
  std::vector<tracker::LocalObject>
      dead_tracking_object;  // TOOD inefficient implementation in here
  for (auto lo = local_objects_list.begin(); lo < local_objects_list.end();) {
    // two criterion to determine whether tracking failure occurs:
    // 1. too long from the last update by detector
    // 2. continuous tracking failure in optical flow tracking
    if (lo->tracking_fail_count >= track_fail_timeout_tick ||
        lo->detector_update_count >= detector_update_timeout_tick) {
      dead_tracking_object.push_back(*lo);
      std::lock_guard<std::mutex> lk(mtx);  // lock the thread
      lo = local_objects_list.erase(lo);
      continue;
    } else {
      // also disable opt when the occulusion occurs
      if (lo->detector_update_count >= stop_opt_timeout) {
        lo->is_opt_enable = false;
      }
      lo++;
    }
  }
  return dead_tracking_object;
}

void MotTracker::report_local_object() {
  // lock_guard<mutex> lk(mtx); //lock the thread
  AINFO_F("------Local Object List Summary------");
  AINFO_F("Local Object Num: {}", local_objects_list.size());
  for (auto lo : local_objects_list) {
    AINFO_F("id: {} | database images num: {}", lo.id, lo.img_blocks.size());
    std::cout << lo.bbox << std::endl;
  }
  AINFO_F("------Summary End------");
}

void MotTracker::visualize_tracking(cv::Mat &img) {
  // lock_guard<mutex> lk(mtx); //lock the thread
  for (auto lo : local_objects_list) {
    if (lo.is_opt_enable) {
      cv::rectangle(img, lo.bbox, lo.color, 4.0);
      cv::rectangle(img, cv::Rect2d(lo.bbox.x, lo.bbox.y, 40, 15), lo.color,
                    -1);
      cv::putText(img, "id:" + std::to_string(lo.id),
                  cv::Point(lo.bbox.x, lo.bbox.y + 15),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // for (auto kp : lo.keypoints_pre)
    //     cv::circle(img, kp, 2, cv::Scalar(255, 0, 0), 2);
  }
  // m_track_vis_pub.publish(
  //     cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg());
}

void MotTracker::detector_and_tracker_association(
    const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
    const std::vector<Eigen::VectorXf> &features,
    std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec) {
  // lock_guard<mutex> lk(mtx); //lock
  if (!local_objects_list.empty()) {
    AINFO_F("SUMMARY: {} bboxes detected!", bboxes.size());
    // perform matching from detector to tracker
    for (int i = 0; i < bboxes.size(); i++) {
      tracker::AssociationVector one_detected_object_ass_vec;
      cv::Rect2d detector_bbox =
          tracker::BboxPadding(bboxes[i], block_max, detector_bbox_padding);
      AINFO << "Detector" << i << " bbox: " << bboxes[i] << std::endl;
      /*Data Association part:
        - two critertion:
          - augemented bboxes(bbox with padding) have enough overlap
          - Reid score is higher than a certain value
      */
      for (int j = 0; j < local_objects_list.size(); j++) {
        double bbox_overlap_ratio = tracker::cal_bbox_overlap_ratio(
            local_objects_list[j].bbox, detector_bbox);
        AINFO_F("Bbox overlap ratio: {}", bbox_overlap_ratio);
        AINFO << "Tracker " << local_objects_list[j].id
              << " bbox: " << local_objects_list[j].bbox;
        if (bbox_overlap_ratio > bbox_overlap_ratio_threshold) {
          // TODO might speed up in here
          float min_query_score =
              local_objects_list[j].find_min_query_score(features[i]);

          // find a match, add it to association vector to construct the
          // association graph
          if (min_query_score < reid_match_threshold)
            one_detected_object_ass_vec.add(tracker::AssociationType(
                j, min_query_score,
                tracker::cal_bbox_match_score(bboxes[i],
                                              local_objects_list[j].bbox)));
        }
      }
      if (one_detected_object_ass_vec.ass_vector.size() > 1)
        one_detected_object_ass_vec.reranking();
      one_detected_object_ass_vec.report();
      AINFO_F("---------------------------------");
      all_detected_bbox_ass_vec.push_back(one_detected_object_ass_vec);
    }
    uniquify_detector_association_vectors(all_detected_bbox_ass_vec,
                                          local_objects_list.size());

    AINFO_F("---Report after uniquification---");
    for (auto ass : all_detected_bbox_ass_vec) {
      ass.report();
    }
    AINFO_F("---Report finished---");
  } else {
    // create empty association vectors to indicate all the detected objects are
    // new
    all_detected_bbox_ass_vec = std::vector<tracker::AssociationVector>(
        bboxes.size(), tracker::AssociationVector());
  }
}

void MotTracker::manage_local_objects_list_by_detector(
    const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
    const std::vector<Eigen::VectorXf> &features, const cv::Mat &img,
    const ros::Time &update_time,
    const std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec) {
  for (int i = 0; i < all_detected_bbox_ass_vec.size(); i++) {
    if (all_detected_bbox_ass_vec[i].ass_vector.empty()) {
      // this detected object is a new object
      AINFO_F("Adding Tracking Object with ID: {}", local_id_not_assigned);
      cv::Mat example_img;
      cv::resize(img(bboxes[i]), example_img,
                 cv::Size(128, 256));  // hard code in here
      tracker::LocalObject new_object(local_id_not_assigned, bboxes[i],
                                      kf_param, update_time, features[i],
                                      example_img);
      local_id_not_assigned++;
      // update database
      update_local_database(new_object, img(new_object.bbox));
      std::lock_guard<std::mutex> lk(mtx);  // lock the thread
      local_objects_list.push_back(new_object);
    } else {
      // re-detect a previous tracking object
      int matched_id = all_detected_bbox_ass_vec[i].ass_vector[0].id;
      AINFO_F("Object {} re-detected!", local_objects_list[matched_id].id);

      local_objects_list[matched_id].track_bbox_by_detector(bboxes[i],
                                                            update_time);
      local_objects_list[matched_id].features.push_back(features[i]);
      local_objects_list[matched_id].update_feat(features[i],
                                                 feature_smooth_ratio);

      // update database
      // TODO this part can be removed later
      update_local_database(
          local_objects_list[matched_id],
          img(local_objects_list[matched_id].bbox & block_max));
    }
  }

  // rectify the bbox
  bbox_rect(block_max);
}

void MotTracker::manage_local_objects_list_by_reid_detector(
    const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
    const std::vector<Eigen::VectorXf> &feat_eigen,
    const std::vector<float> &feat_vector, const cv::Mat &img,
    const ros::Time &update_time,
    const std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec) {
  const int feat_dimension = feat_vector.size() / feat_eigen.size();

  for (int i = 0; i < all_detected_bbox_ass_vec.size(); i++) {
    if (all_detected_bbox_ass_vec[i].ass_vector.empty()) {
      // this detected object is a new object
      AINFO_F("Adding Tracking Object with ID: {}", local_id_not_assigned);
      cv::Mat example_img;
      cv::resize(img(bboxes[i]), example_img,
                 cv::Size(128, 256));  // hard code in here
      tracker::LocalObject new_object(local_id_not_assigned, bboxes[i],
                                      kf_param, update_time, feat_eigen[i],
                                      example_img);
      local_id_not_assigned++;
      std::lock_guard<std::mutex> lk(mtx);  // lock the thread
      // insert the 2048d feature vector
      new_object.features_vector.insert(new_object.features_vector.end(),
                                        feat_vector.begin(), feat_vector.end());

      local_objects_list.push_back(new_object);
    } else {
      // re-detect a previous tracking object
      int matched_id = all_detected_bbox_ass_vec[i].ass_vector[0].id;
      AINFO_F("Object {} re-detected!", local_objects_list[matched_id].id);

      local_objects_list[matched_id].track_bbox_by_detector(bboxes[i],
                                                            update_time);
      local_objects_list[matched_id].update_feat(feat_eigen[i],
                                                 feature_smooth_ratio);
      // insert the 2048d feature vector
      local_objects_list[matched_id].features_vector.insert(
          local_objects_list[matched_id].features_vector.end(),
          feat_vector.begin() + i * feat_dimension,
          feat_vector.begin() + (i + 1) * feat_dimension);
    }
  }

  // rectify the bbox
  bbox_rect(block_max);
}

std::vector<cv::Rect2d> MotTracker::bbox_ros_to_opencv(
    const std::vector<std_msgs::UInt16MultiArray> &bbox_ros) {
  std::vector<cv::Rect2d> bbox_opencv;
  for (auto b : bbox_ros) {
    bbox_opencv.push_back(
        cv::Rect2d(b.data[0], b.data[1], b.data[2], b.data[3]));
  }
  return bbox_opencv;
}

void MotTracker::bbox_rect(const cv::Rect2d &bbox_max) {
  for (auto &lo : local_objects_list) {
    // std::cout << lo.bbox << std::endl;
    // std::cout << bbox_max << std::endl;
    lo.bbox = lo.bbox & bbox_max;
    // std::cout << lo.bbox << std::endl;
  }
}

/** Refactored **/

bool MotTracker::Track(const cv::Mat &frame, const ros::Time &timestamp) {
  ComputeFlow(frame);
  ApplyKalman(frame, timestamp);
  // TODO is it necessary to remvoe dead tracks in this step?
  return true;
}

bool MotTracker::Track(const cv::Mat &frame, const ros::Time &timestamp,
                       const ObjectDetectionResults &detections) {
  ComputeFlow(frame);
  ApplyKalman(frame, timestamp);

  cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(frame.cols, frame.rows));
  std::vector<tracker::AssociationVector> all_detected_bbox_ass_vec;
  if (!local_objects_list.empty()) {
    // perform matching from detector to tracker
    for (size_t i = 0; i < detections.size(); i++) {
      tracker::AssociationVector one_detected_object_ass_vec;
      cv::Rect2d detector_bbox = tracker::BboxPadding(
          detections[i].bbox, block_max, detector_bbox_padding);
      /*Data Association part:
        - two critertion:
          - augemented bboxes(bbox with padding) have enough overlap
          - Reid score is higher than a certain value
      */
      for (size_t j = 0; j < local_objects_list.size(); j++) {
        double bbox_overlap_ratio = tracker::cal_bbox_overlap_ratio(
            local_objects_list[j].bbox, detector_bbox);
        if (bbox_overlap_ratio > bbox_overlap_ratio_threshold) {
          one_detected_object_ass_vec.add(tracker::AssociationType(
              j, bbox_overlap_ratio,
              tracker::cal_bbox_match_score(detections[i].bbox,
                                            local_objects_list[j].bbox)));
        }
      }
      if (one_detected_object_ass_vec.ass_vector.size() > 1)
        one_detected_object_ass_vec.reranking();
      // one_detected_object_ass_vec.report();
      // AINFO_F("---------------------------------");
      all_detected_bbox_ass_vec.push_back(one_detected_object_ass_vec);
    }
    uniquify_detector_association_vectors(all_detected_bbox_ass_vec,
                                          local_objects_list.size());
  } else {
    // create empty association vectors to indicate all the detected objects are
    // new
    all_detected_bbox_ass_vec = std::vector<tracker::AssociationVector>(
        detections.size(), tracker::AssociationVector());
  }

  // update matched tracks
  for (size_t i = 0; i < all_detected_bbox_ass_vec.size(); i++) {
    if (all_detected_bbox_ass_vec[i].ass_vector.empty()) {
      // this detected object is a new object
      AINFO_F("Adding Tracking Object with ID: {}", local_id_not_assigned);
      tracker::LocalObject new_object(local_id_not_assigned, detections[i].bbox,
                                      kf_param, timestamp);
      local_id_not_assigned++;
      local_objects_list.push_back(new_object);
    } else {
      // re-detect a previous tracking object
      int matched_id = all_detected_bbox_ass_vec[i].ass_vector[0].id;
      AINFO_F("Object {} re-detected!", local_objects_list[matched_id].id);
      local_objects_list[matched_id].track_bbox_by_detector(detections[i].bbox,
                                                            timestamp);
    }
  }

  // rectify the bbox
  bbox_rect(block_max);

  // remove dead tracks
  auto dead_tracker = remove_dead_trackers();

  return true;
}

bool MotTracker::Track(const cv::Mat &frame, const ros::Time &timestamp,
                       const ObjectDetectionResults &detections,
                       const std::vector<float> embeddings) {
  return true;
}

void MotTracker::ComputeFlow(const cv::Mat &frame) {
  // TODO use list comprehension to filter active tracks
  // get the bbox measurement by optical flow
  auto ret = opt_tracker.update(frame, local_objects_list);
  // clear tracks when camera motion cannot be estimated
  if (!ret) {
    local_objects_list.clear();
  }
}

void MotTracker::ApplyKalman(const cv::Mat &frame, const ros::Time &timestamp) {
  // TODO give large flow uncertainty for occluded tracks, usually these with
  // high age and low inlier ratio

  cv::Rect2d block_max(cv::Point2d(0, 0), cv::Point2d(frame.cols, frame.rows));

  for (auto &lo : local_objects_list) {
    // update each tracking object in tracking list by kalman filter
    lo.track_bbox_by_optical_flow(timestamp);
    lo.bbox = lo.bbox & block_max;

    // TODO mark out tracks as lost which have low IoM
  }

  // std::cout << lo.bbox << std::endl;
}

}  // namespace inference
}  // namespace zetton