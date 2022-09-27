#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Convert the input image from BGR to RGB
class BGR2RGB : public BaseTransform {
 public:
  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "BGR2RGB"; }

  /// \brief transform the input image
  static bool Run(Mat* mat,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

/// \brief Convert the input image from RGB to BGR
class RGB2BGR : public BaseTransform {
 public:
  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "RGB2BGR"; }

  /// \brief transform the input image
  static bool Run(Mat* mat,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
