#include <zetton_inference/tracker/mot/optical_flow.h>

namespace zetton {
namespace inference {
namespace tracker {

bool OpticalFlow::update(const cv::Mat &frame_curr_bgr,
                         std::vector<LocalObject> &local_objects) {
  // preprocess frame
  cv::Mat frame_curr;
  if (optical_flow_param_.use_resize) {
    cv::resize(frame_curr_bgr, frame_curr, cv::Size(),
               1.0 / optical_flow_param_.resize_factor,
               1.0 / optical_flow_param_.resize_factor, cv::INTER_AREA);
    cv::cvtColor(frame_curr, frame_curr, cv::COLOR_BGR2GRAY);
  } else {
    cv::cvtColor(frame_curr_bgr, frame_curr, cv::COLOR_BGR2GRAY);
  }

  // initialize with frame
  if (frame_pre_.empty() || local_objects.empty()) {
    frame_pre_ = frame_curr;
    for (auto &lo : local_objects) {
      lo.is_track_succeed = false;
    }
    return true;
  }

  // detect enough keypoints for each tracking object,
  // and forms a keypoints vector that conatins all the keypoints
  std::vector<cv::Point2f> keypoints_pre_all, keypoints_curr_all;
  detect_enough_keypoints(local_objects, keypoints_pre_all);

  // track keypoints by optical flow
  std::vector<uchar> status;  // tracking succeed or not
  std::vector<float> errors;  // tracking error
  if (!keypoints_pre_all.empty()) {
    // match features using optical flow
    cv::calcOpticalFlowPyrLK(frame_pre_, frame_curr, keypoints_pre_all,
                             keypoints_curr_all, status, errors);
    // for (int i = 0; i < keypoints_curr_all.size(); i++)
    // {
    //     std::cout << keypoints_pre_all[i] << std::endl;
    //     std::cout << keypoints_curr_all[i] << std::endl;
    // }

    // calculate homography matrix to compensate camera motion
    camera_motion_compensate(keypoints_curr_all, status);

    // remove the keypoints that fails to track,
    // and update the current keypoints of the local object
    update_local_objects_curr_kp(local_objects, keypoints_curr_all, status);

    // calculate the transform matrix and remove the outliers
    calculate_measurement(local_objects);

    // update variable
    frame_pre_ = frame_curr;
    for (auto &lo : local_objects) {
      lo.keypoints_pre = lo.keypoints_curr;
    }
  } else {
    frame_pre_ = frame_curr;
    for (auto &lo : local_objects) {
      lo.is_track_succeed = false;
    }
  }

  return is_camera_motion_estimated;
}

void OpticalFlow::detect_enough_keypoints(
    std::vector<LocalObject> &local_objects,
    std::vector<cv::Point2f> &keypoints_all) {
  for (auto &lo : local_objects) {
    // detect fast keypoints for the objects with too few succefully tracked
    // keypoints
    if (lo.keypoints_pre.size() < optical_flow_param_.min_keypoints_to_track *
                                      min_keypoints_num_factor(lo.bbox)) {
      if (optical_flow_param_.use_resize) {
        cv::Rect2d bbox_resize = cv::Rect2d(
            1.0 * lo.bbox.x / optical_flow_param_.resize_factor,
            1.0 * lo.bbox.y / optical_flow_param_.resize_factor,
            1.0 * lo.bbox.width / optical_flow_param_.resize_factor,
            1.0 * lo.bbox.height / optical_flow_param_.resize_factor);
        cv::goodFeaturesToTrack(
            frame_pre_(bbox_resize), lo.keypoints_pre,
            optical_flow_param_.corner_detector_max_num,
            optical_flow_param_.corner_detector_quality_level,
            optical_flow_param_.corner_detector_min_distance, cv::noArray(),
            optical_flow_param_.corner_detector_block_size,
            optical_flow_param_.corner_detector_use_harris,
            optical_flow_param_.corner_detector_k);
      } else {
        cv::goodFeaturesToTrack(
            frame_pre_(lo.bbox), lo.keypoints_pre,
            optical_flow_param_.corner_detector_max_num,
            optical_flow_param_.corner_detector_quality_level,
            optical_flow_param_.corner_detector_min_distance, cv::noArray(),
            optical_flow_param_.corner_detector_block_size,
            optical_flow_param_.corner_detector_use_harris,
            optical_flow_param_.corner_detector_k);
      }

      // add  offset
      cv::Point2f tmp(1.0 * lo.bbox.x / optical_flow_param_.resize_factor,
                      1.0 * lo.bbox.y / optical_flow_param_.resize_factor);
      for (auto &p : lo.keypoints_pre) {
        p = p + tmp;
      }
    }
    keypoints_all.insert(keypoints_all.end(), lo.keypoints_pre.begin(),
                         lo.keypoints_pre.end());
  }

  // detect the keypoints for tracking the motion of the camera
  if (keypoints_vo_pre.size() <
      optical_flow_param_.min_keypoints_for_motion_estimation) {
    cv::goodFeaturesToTrack(frame_pre_, keypoints_vo_pre,
                            optical_flow_param_.corner_detector_max_num,
                            optical_flow_param_.corner_detector_quality_level,
                            optical_flow_param_.corner_detector_min_distance,
                            cv::noArray(),
                            optical_flow_param_.corner_detector_block_size,
                            optical_flow_param_.corner_detector_use_harris,
                            optical_flow_param_.corner_detector_k);
  }
  keypoints_all.insert(keypoints_all.end(), keypoints_vo_pre.begin(),
                       keypoints_vo_pre.end());
  // std::cout << keypoints_all.size() << std::endl;
}

void OpticalFlow::camera_motion_compensate(
    std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status) {
  // deal with keypoints used to calculate the Homography matrix between two
  // frame
  keypoints_vo_curr.clear();
  for (int i = keypoints_all.size() - keypoints_vo_pre.size();
       i < keypoints_all.size(); i++) {
    if (int(status[i]) == 1) {
      keypoints_vo_curr.push_back(keypoints_all[i]);
    } else {
      keypoints_vo_pre.erase(keypoints_vo_pre.begin() +
                             keypoints_vo_curr.size());
    }
  }

  std::vector<uchar> inliers;
  // if the successfully tracked keypoints is too few, we takes it as a failure
  // in tracking
  if (keypoints_vo_curr.size() <
      optical_flow_param_.min_keypoints_to_cal_H_mat) {
    std::cout << "Too few points for estimating camera motion, estimate homography matrix fails..."
              << std::endl;
    is_motion_estimation_succeeed = false;
    keypoints_vo_curr.clear();
    return;
  } else {
    H_motion = cv::findHomography(keypoints_vo_pre, keypoints_vo_curr, inliers,
                                  cv::RANSAC);
    // empty matrix means fail to estimate the transform matrix
    if (H_motion.empty()) {
      std::cout << "Estimate homography matrix fails..." << std::endl;
      is_motion_estimation_succeeed = false;
      is_camera_motion_estimated = false;
      return;
    } else {  // estimate succeed
      is_motion_estimation_succeeed = true;
      is_camera_motion_estimated = true;
      // get transformed bbox
      // std::cout << "Motion compensation matrix" << H_motion << std::endl;

      // remove outliers
      int i = 0;
      for (auto kpc = keypoints_vo_curr.begin();
           kpc != keypoints_vo_curr.end();) {
        // outlier detected
        if (int(inliers[i]) == 0) {
          kpc = keypoints_vo_curr.erase(kpc);  // remove outlier
        } else {
          kpc++;
        }
        i++;
      }

      // update the motion estimation keypoints
      keypoints_vo_pre = keypoints_vo_curr;
    }
  }
}

void OpticalFlow::update_local_objects_curr_kp(
    std::vector<LocalObject> &local_objects,
    std::vector<cv::Point2f> &keypoints_all, const std::vector<uchar> &status) {
  int start = 0, end = 0, i;
  for (auto &lo : local_objects) {
    lo.keypoints_curr.clear();
    end += lo.keypoints_pre.size();
    for (i = start; i < end; i++) {
      if (int(status[i]) == 1) {
        if (is_motion_estimation_succeeed) {
          if (!is_scene_points(lo.keypoints_pre[lo.keypoints_curr.size()],
                               keypoints_all[i])) {
            lo.keypoints_curr.push_back(keypoints_all[i]);
          } else {
            lo.keypoints_pre.erase(lo.keypoints_pre.begin() +
                                   lo.keypoints_curr.size());
          }
        } else {
          lo.keypoints_curr.push_back(keypoints_all[i]);
        }
      } else {
        lo.keypoints_pre.erase(lo.keypoints_pre.begin() +
                               lo.keypoints_curr.size());
      }
    }
    start = end;
  }
}

void OpticalFlow::calculate_measurement(
    std::vector<LocalObject> &local_objects) {
  for (auto &lo : local_objects) {
    std::vector<uchar> inliers;
    // if the successfully tracked keypoints is too few, we takes it as a
    // failure in tracking
    if (lo.keypoints_curr.size() <
        optical_flow_param_.min_keypoints_to_cal_H_mat) {
      std::cout << "Too few point for calculating measurement, estimate affine partial matrix fails..."
                << std::endl;
      lo.is_track_succeed = false;
      lo.keypoints_curr.clear();  // clear all the points
      continue;
    } else {
      // std::cout << lo.keypoints_pre.size() << std::endl;
      // std::cout << lo.keypoints_curr.size() << std::endl;
      cv::Mat H = cv::estimateAffinePartial2D(
          lo.keypoints_pre, lo.keypoints_curr, inliers, cv::RANSAC);
      // empty matrix means fail to estimate the transform matrix
      if (H.empty()) {
        std::cout << "Estimate affine partial matrix fails..." << std::endl;
        lo.is_track_succeed = false;
        continue;
      }
      // estimate succeed
      else {
        lo.is_track_succeed = true;
        // get transformed bbox

        if (optical_flow_param_.use_resize) {
          H.at<double>(0, 2) =
              H.at<double>(0, 2) * optical_flow_param_.resize_factor;
          H.at<double>(1, 2) =
              H.at<double>(1, 2) * optical_flow_param_.resize_factor;
        }
        lo.T_measurement = H;
        // std::cout << H << std::endl;

        // remove outliers
        int i = 0;
        for (auto lo_kpc = lo.keypoints_curr.begin();
             lo_kpc != lo.keypoints_curr.end();) {
          // outlier detected
          if (int(inliers[i]) == 0) {
            lo_kpc = lo.keypoints_curr.erase(lo_kpc);  // remove outlier
          } else {
            lo_kpc++;
          }
          i++;
        }
      }
    }
  }
}

inline double OpticalFlow::min_keypoints_num_factor(const cv::Rect2d &bbox) {
  return (bbox.area() / optical_flow_param_.keypoints_num_factor_area);
}

inline cv::Rect2d OpticalFlow::transform_bbox(const cv::Mat &H,
                                              const cv::Rect2d &bbox_pre) {
  double x1, y1, x2, y2;
  x1 = bbox_pre.tl().x * H.at<double>(0, 0) +
       bbox_pre.tl().y * H.at<double>(0, 1) + H.at<double>(0, 2);
  x2 = bbox_pre.br().x * H.at<double>(0, 0) +
       bbox_pre.br().y * H.at<double>(0, 1) + H.at<double>(0, 2);
  y1 = bbox_pre.tl().x * H.at<double>(1, 0) +
       bbox_pre.tl().y * H.at<double>(1, 1) + H.at<double>(1, 2);
  y2 = bbox_pre.br().x * H.at<double>(1, 0) +
       bbox_pre.br().y * H.at<double>(1, 1) + H.at<double>(1, 2);
  return cv::Rect2d(cv::Point2d(x1, y1), cv::Point2d(x2, y2));
}

inline bool OpticalFlow::is_scene_points(const cv::Point2f &kp_pre,
                                         const cv::Point2f &kp_curr) {
  double result = pow(kp_pre.x * H_motion.at<double>(0, 0) +
                          kp_pre.y * H_motion.at<double>(0, 1) +
                          H_motion.at<double>(0, 2) - kp_curr.x,
                      2) +
                  pow(kp_pre.x * H_motion.at<double>(1, 0) +
                          kp_pre.y * H_motion.at<double>(1, 1) +
                          H_motion.at<double>(1, 2) - kp_curr.y,
                      2);
  // std::cout << result << std::endl;
  // std::cout << H_motion << std::endl;
  // std::cout << kp_pre << std::endl;
  // std::cout << kp_curr << std::endl;
  if (result < optical_flow_param_.min_pixel_dis_square_for_scene_point)
    return true;
  else
    return false;
}

}  // namespace tracker
}  // namespace inference
}  // namespace zetton