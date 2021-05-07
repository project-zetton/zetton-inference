#pragma once

#include <std_msgs/UInt16MultiArray.h>

#include "zetton_inference/interface/base_object_detector.h"
#include "zetton_inference/interface/base_object_tracker.h"
#include "zetton_inference/tracker/util/association_type.h"
#include "zetton_inference/tracker/util/kalman_filter.h"
#include "zetton_inference/tracker/util/local_object.h"
#include "zetton_inference/tracker/util/optical_flow.h"
#include "zetton_inference/tracker/util/util.h"

namespace zetton {
namespace inference {

class MotTracker : public BaseObjectTracker {
 public:
  MotTracker() = default;
  ~MotTracker() override = default;

  bool Init() override;
  void Infer() override{};

  bool Track() override { return true; };
  bool Track(const cv::Mat &frame, const ros::Time &timestamp);
  bool Track(const cv::Mat &frame, const ros::Time &timestamp,
             const ObjectDetectionResults &detections);
  bool Track(const cv::Mat &frame, const ros::Time &timestamp,
             const ObjectDetectionResults &detections,
             const std::vector<float> embeddings);

  std::vector<tracker::LocalObject> &tracks() { return local_objects_list; };

 private:
  void ComputeFlow(const cv::Mat &frame);
  void ApplyKalman(const cv::Mat &frame, const ros::Time &timestamp);

 private:
  std::vector<tracker::LocalObject> update_bbox_by_tracker(
      const cv::Mat &img, const ros::Time &update_time);
  void update_bbox_by_detector(const cv::Mat &img,
                               const std::vector<cv::Rect2d> &bboxes,
                               const std::vector<float> feature,
                               const ros::Time &update_time);

 private:
  void load_config(ros::NodeHandle *n);

  bool update_local_database(tracker::LocalObject &local_object,
                             const cv::Mat &img_block);

  void update_overlap_flag();

  // bbox update by optical flow tracker
  void track_bbox_by_optical_flow(const cv::Mat &img,
                                  const ros::Time &update_time,
                                  bool update_database);
  std::vector<tracker::LocalObject> remove_dead_trackers();
  void report_local_object();
  void visualize_tracking(cv::Mat &img);

  // associate the detected results with local tracking objects, make sure one
  // detected object matches only 0 or 1 tracking object
  void detector_and_tracker_association(
      const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
      const std::vector<Eigen::VectorXf> &features,
      std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec);
  void manage_local_objects_list_by_detector(
      const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
      const std::vector<Eigen::VectorXf> &features, const cv::Mat &img,
      const ros::Time &update_time,
      const std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec);

  void manage_local_objects_list_by_reid_detector(
      const std::vector<cv::Rect2d> &bboxes, const cv::Rect2d &block_max,
      const std::vector<Eigen::VectorXf> &feat_eigen,
      const std::vector<float> &feat_vector, const cv::Mat &img,
      const ros::Time &update_time,
      const std::vector<tracker::AssociationVector> &all_detected_bbox_ass_vec);

  std::vector<cv::Rect2d> bbox_ros_to_opencv(
      const std::vector<std_msgs::UInt16MultiArray> &bbox_ros);
  void bbox_rect(const cv::Rect2d &bbox_max);

 private:
  std::vector<tracker::LocalObject> local_objects_list;

  ros::NodeHandle nh_;

  // lock
  std::mutex mtx;

  tracker::OpticalFlow opt_tracker;
  int local_id_not_assigned = 0;

  // params
  int track_fail_timeout_tick = 30;
  double bbox_overlap_ratio_threshold = 0.5;
  int track_to_reid_bbox_margin = 10;
  float height_width_ratio_min = 1.0;
  float height_width_ratio_max = 3.0;
  float record_interval = 0.1;

  int detector_update_timeout_tick = 10;
  int stop_opt_timeout = 5;
  int detector_bbox_padding = 10;
  float reid_match_threshold = 200;
  double reid_match_bbox_dis = 30;
  double reid_match_bbox_size_diff = 30;
  int match_centroid_padding = 20;
  float feature_smooth_ratio = 0.8;

  // params
  tracker::OpticalFlowParam opt_param;
  tracker::KalmanFilterParam kf_param;
};

}  // namespace inference
}  // namespace zetton