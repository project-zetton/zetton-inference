#include "zetton_inference/vision/base/transform/pad.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/base/matrix.h"
#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

bool Pad::RunOnOpenCV(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("Invalid input format: {}", ToString(mat->layout));
    return false;
  }
  if (mat->Channels() > 4) {
    AERROR_F("Invalid input channels: {}", mat->Channels());
    return false;
  }
  if (mat->Channels() != static_cast<int>(value_.size())) {
    AERROR_F(
        "The number of values ({}) is not equal to the number of channels ({})",
        value_.size(), mat->Channels());
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  cv::copyMakeBorder(*im, *im, top_, bottom_, left_, right_,
                     cv::BORDER_CONSTANT, value);
  mat->SetHeight(im->rows);
  mat->SetWidth(im->cols);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Pad::GpuRun(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("Invalid input format: {}", ToString(mat->layout));
    return false;
  }
  if (mat->Channels() > 4) {
    AERROR_F("Invalid input channels: {}", mat->Channels());
    return false;
  }
  if (mat->Channels() != value_.size()) {
    AERROR_F(
        "The number of values ({}) is not equal to the number of channels ({})",
        value_.size(), mat->Channels());
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  cv::cuda::copyMakeBorder(*im, *im, top_, bottom_, left_, right_,
                           cv::BORDER_CONSTANT, value);
  mat->SetHeight(im->rows);
  mat->SetWidth(im->cols);
  return true;
}
#endif

bool Pad::Run(Mat* mat, const int& top, const int& bottom, const int& left,
              const int& right, const std::vector<float>& value,
              TransformLibraryType lib) {
  auto p = Pad(top, bottom, left, right, value);
  return p(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
