#pragma once

#include <Eigen/Dense>

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

/// \brief Kalman filter for byte tracker
class KalmanFilter {
 public:
  using DetectBox = std::array<float, 4>;
  using StateMean = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
  using StateCov = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
  using StateHMean = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
  using StateHCov = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

 public:
  /// \brief constructor
  /// \param std_weight_position standard deviation of position weight
  /// \param std_weight_velocity standard deviation of velocity weight
  KalmanFilter(const float& std_weight_position = 1. / 20,
               const float& std_weight_velocity = 1. / 160);

 public:
  /// \brief initialize the state of kalman filter
  /// \param mean state mean of the kalman filter
  /// \param covariance state covariance of the kalman filter
  /// \param measurement input detection results as measurement
  void Init(StateMean& mean, StateCov& covariance,
            const DetectBox& measurement);

  /// \brief predict the state of kalman filter in the next frame
  void Predict(StateMean& mean, StateCov& covariance);

  /// \brief update the state of kalman filter with the given measurement
  /// \param mean state mean of the kalman filter
  /// \param covariance state covariance of the kalman filter
  /// \param measurement input detection results as measurement
  void Update(StateMean& mean, StateCov& covariance,
              const DetectBox& measurement);

 private:
  /// \brief project the state of kalman filter to the measurement space
  /// \param projected_mean projected state mean of the kalman filter
  /// \param projected_covariance projected state covariance of the kalman
  /// filter
  /// \param mean state mean of the kalman filter
  /// \param covariance state covariance of the kalman filter
  void Project(StateHMean& projected_mean, StateHCov& projected_covariance,
               const StateMean& mean, const StateCov& covariance);

 private:
  /// \brief standard deviation of position weight
  float std_weight_position_;
  /// \brief standard deviation of velocity weight
  float std_weight_velocity_;

  /// \brief state transition matrix of the kalman filter
  Eigen::Matrix<float, 8, 8, Eigen::RowMajor> motion_mat_;
  /// \brief measurement matrix of the kalman filter
  Eigen::Matrix<float, 4, 8, Eigen::RowMajor> update_mat_;
};

}  // namespace bytetrack
}  // namespace vision
}  // namespace inference
}  // namespace zetton
