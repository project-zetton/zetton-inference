#include "zetton_inference/vision/base/transform/pad.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/base/matrix.h"
#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

bool Pad::RunOnOpenCV(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("Pad: The input data must be Layout::HWC format!");
    return false;
  }
  if (mat->Channels() > 4) {
    AERROR_F("Pad: Only support channels <= 4.");
    return false;
  }
  if (mat->Channels() != value_.size()) {
    AERROR_F(
        "Pad: Require input channels equals to size of padding value, "
        "but now channels = {}, the size of padding values = {}.",
        mat->Channels(), value_.size());
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
    AERROR_F("Pad: The input data must be Layout::HWC format!");
    return false;
  }
  if (mat->Channels() > 4) {
    AERROR_F("Pad: Only support channels <= 4.");
    return false;
  }
  if (mat->Channels() != value_.size()) {
    AERROR_F(
        "Pad: Require input channels equals to size of padding value, "
        "but now channels = {}, the size of padding values = {}.",
        mat->Channels(), value_.size());
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
