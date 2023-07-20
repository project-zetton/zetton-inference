#pragma once

#include <string>

#include "zetton_inference/vision/common/matrix.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief External library interface for image transformation
enum TransformLibraryType { kDefault, kOpenCV, kOpenCVCUDA };

/// \brief Base class for image transformation
class BaseTransform {
 public:
  /// \brief default library to be used
  static TransformLibraryType default_lib;

  /// \brief name of the transform
  virtual std::string Name() = 0;
  /// \brief transform the input image on CPU
  virtual bool RunOnOpenCV(Mat* mat);
#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  virtual bool RunOnOpenCVCUDA(Mat* mat);
#endif

  /// \brief transform the input image
  /// \param mat input image
  /// \param lib transform library to be used
  virtual bool operator()(
      Mat* mat, TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
