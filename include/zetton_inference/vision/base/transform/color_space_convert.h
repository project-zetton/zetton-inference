#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

class BGR2RGB : public BaseTransform {
 public:
  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif
  std::string Name() override { return "BGR2RGB"; }

  static bool Run(Mat* mat,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

class RGB2BGR : public BaseTransform {
 public:
  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif
  std::string Name() override { return "RGB2BGR"; }

  static bool Run(Mat* mat,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
