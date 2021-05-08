#pragma once

#include <geometry_msgs/Point.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "zetton_inference/tracker/mot/kalman_filter.h"
#include "zetton_inference/tracker/mot/timer.h"
#include "zetton_inference/tracker/mot/util.h"

namespace zetton {
namespace inference {
namespace tracker {
class LocalObject {
 public:
  LocalObject(const int id_init, const cv::Rect2d &bbox_init,
              const KalmanFilterParam &kf_param_init, const ros::Time &time_now,
              const Eigen::VectorXf &feat = {},
              const cv::Mat &example_image = {});

  // update bbox by optical flow
  void track_bbox_by_optical_flow(const ros::Time &time_now);

  // update bbox by detector
  // general before track by detector, we will perform track by optical flow
  // first
  void track_bbox_by_detector(const cv::Rect2d &bbox_detector,
                              const ros::Time &update_time);

  float find_min_query_score(const Eigen::VectorXf &query);

  cv::Rect2d bbox_of_lidar_time(const ros::Time &time_now);

  void update_feat(const Eigen::VectorXf &feature_new,
                   float smooth_ratio = 0.7);

  inline void Draw(cv::Mat &frame) {
    cv::rectangle(frame, bbox, color, 2);
    std::stringstream stream;
    stream << "Track " << id;
    cv::putText(frame, stream.str(), cv::Point(bbox.x, bbox.y - 5), 0, 0.5,
                color, 2);
  }

  inline friend std::ostream &operator<<(std::ostream &os,
                                         const LocalObject &track) {
    os << track.id << ":" << track.bbox;
    return os;
  }

  int id;
  bool is_opt_enable = true;
  cv::Rect2d bbox;
  cv::Mat T_measurement;

  // state time tick count
  int tracking_fail_count = 0;
  int detector_update_count = 0;
  int overlap_count = 0;

  cv::Scalar color;
  bool is_track_succeed;
  std::vector<cv::Mat> img_blocks;

  std::vector<Eigen::VectorXf> features;
  Eigen::VectorXf features_now;
  std::vector<float> features_vector;
  geometry_msgs::Point position;

  cv::Mat example_image;

  ros::Time bbox_last_update_time;
  ros::Time ros_time_pc_last;
  bool is_overlap = false;

  // control the time interval between adding two image blocks into this local
  // object database
  tracker::Timer database_update_timer;

  std::vector<cv::Point2f> keypoints_pre, keypoints_curr;
  cv::Rect2d bbox_measurement;

 private:
  KalmanFilter *kf;
};

}  // namespace tracker
}  // namespace inference
}  // namespace zetton