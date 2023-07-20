#include "zetton_inference/vision/common/transform/base.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {
namespace vision {

TransformLibraryType BaseTransform::default_lib =
    TransformLibraryType::kDefault;

bool BaseTransform::RunOnOpenCV(Mat* mat) {
  AERROR_F("Transform {} is not implemented on OpenCV", Name());
  return false;
}

#ifdef ENABLE_OPENCV_CUDA
bool BaseTransform::RunOnOpenCVCUDA(Mat* mat) {
  AERROR_F("Transform {} is not implemented on OpenCV CUDA", Name());
  return false;
}
#endif

bool BaseTransform::operator()(Mat* mat, TransformLibraryType lib) {
  // if default_lib is set
  // then use default_lib
  TransformLibraryType target = lib;
  if (default_lib != TransformLibraryType::kDefault) {
    target = default_lib;
  }

  if (target == TransformLibraryType::kOpenCVCUDA) {
#ifdef ENABLE_OPENCV_CUDA
    bool ret = RunOnOpenCVCUDA(mat);
    mat->device = InferenceDeviceType::kGPU;
    return ret;
#else
    AERROR_F("OpenCV CUDA is not enabled");
    return false;
#endif
  }
  bool ret = RunOnOpenCV(mat);
  mat->device = InferenceDeviceType::kCPU;
  return ret;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
