#pragma once

#include <string>

#include "zetton_inference/vision/base/matrix.h"

namespace zetton {
namespace inference {
namespace vision {

enum TransformLibraryType { kDefault, kOpenCV, kOpenCVCUDA };

class BaseTransform {
 public:
  // default_lib has the highest priority
  // all the function in `processor` will force to use
  // default_lib if this flag is set.
  // DEFAULT means this flag is not set
  static TransformLibraryType default_lib;

  virtual std::string Name() = 0;
  virtual bool RunOnOpenCV(Mat* mat);
#ifdef ENABLE_OPENCV_CUDA
  virtual bool RunOnOpenCVCUDA(Mat* mat);
#endif

  virtual bool operator()(
      Mat* mat, TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
