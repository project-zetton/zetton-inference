#include "zetton_inference/vision/base/transform/cast.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

bool Cast::RunOnOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int c = im->channels();
  if (dtype_ == "float") {
    if (im->type() != CV_32FC(c)) {
      im->convertTo(*im, CV_32FC(c));
    }
  } else if (dtype_ == "double") {
    if (im->type() != CV_64FC(c)) {
      im->convertTo(*im, CV_64FC(c));
    }
  } else {
    AWARN_F("Cast not support for {} now! will skip this operation.", dtype_);
  }
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Cast::RunOnOpenCVCUDA(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int c = im->channels();
  if (dtype_ == "float") {
    if (im->type() != CV_32FC(c)) {
      im->convertTo(*im, CV_32FC(c));
    }
  } else if (dtype_ == "double") {
    if (im->type() != CV_64FC(c)) {
      im->convertTo(*im, CV_64FC(c));
    }
  } else {
    AWARN_F("Cast not support for {} now! will skip this operation.", dtype_);
  }
  return true;
}
#endif

bool Cast::Run(Mat* mat, const std::string& dtype, TransformLibraryType lib) {
  auto c = Cast(dtype);
  return c(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
