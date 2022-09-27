#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Convert the input image layout from HWC to CHW
class HWC2CHW : public BaseTransform {
 public:
  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "HWC2CHW"; }

  /// \brief transform the input image
  /// \param mat input image
  /// \param lib transform library to be used
  static bool Run(Mat* mat,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
