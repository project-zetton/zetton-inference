#pragma once

#include <Eigen/Dense>

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

class KalmanFilter {
 public:
  using DetectBox = std::array<float, 4>;
  using StateMean = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
  using StateCov = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
  using StateHMean = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
  using StateHCov = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

 public:
  KalmanFilter(const float& std_weight_position = 1. / 20,
               const float& std_weight_velocity = 1. / 160);

 public:
  void initiate(StateMean& mean, StateCov& covariance,
                const DetectBox& measurement);

  void predict(StateMean& mean, StateCov& covariance);

  void update(StateMean& mean, StateCov& covariance,
              const DetectBox& measurement);

 private:
  float std_weight_position_;
  float std_weight_velocity_;

  Eigen::Matrix<float, 8, 8, Eigen::RowMajor> motion_mat_;
  Eigen::Matrix<float, 4, 8, Eigen::RowMajor> update_mat_;

  void project(StateHMean& projected_mean, StateHCov& projected_covariance,
               const StateMean& mean, const StateCov& covariance);
};

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
