#include "zetton_inference/vision/common/transform/interpolate.h"

#include <opencv2/imgproc.hpp>

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/common/matrix.h"
#include "zetton_inference/vision/common/transform/base.h"

namespace zetton {
namespace inference {
namespace vision {

bool Interpolate::RunOnOpenCV(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("Resize: The format of input is not HWC.");
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    AERROR_F(
        "Invalid resize parameters (width={}, height={}, scale_w={}, "
        "scale_h={})",
        width_, height_, scale_w_, scale_h_);
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Interpolate::RunOnOpenCVCUDA(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("Resize: The format of input is not HWC.");
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::cuda::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::cuda::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::cuda::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    AERROR_F(
        "Invalid resize parameters (width={}, height={}, scale_w={}, "
        "scale_h={})",
        width_, height_, scale_w_, scale_h_);
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}
#endif

bool Interpolate::Run(Mat* mat, int width, int height, float scale_w,
                      float scale_h, int interp, bool use_scale,
                      TransformLibraryType lib) {
  if (mat->Height() == height && mat->Width() == width) {
    return true;
  }
  auto r = Interpolate(width, height, scale_w, scale_h, interp, use_scale);
  return r(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
