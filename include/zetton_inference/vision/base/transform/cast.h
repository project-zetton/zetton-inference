#pragma once

#include <utility>

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Cast the input image to the specified data type
class Cast : public BaseTransform {
 public:
  /// \brief constructor
  explicit Cast(std::string dtype = "float") : dtype_(std::move(dtype)) {}

  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "Cast"; }

  /// \brief transform the input image
  static bool Run(Mat* mat, const std::string& dtype,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  /// \brief data type to be casted
  std::string dtype_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
