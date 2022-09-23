#pragma once

#include <utility>

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

class Cast : public BaseTransform {
 public:
  explicit Cast(std::string dtype = "float") : dtype_(std::move(dtype)) {}
  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif
  std::string Name() override { return "Cast"; }
  static bool Run(Mat* mat, const std::string& dtype,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  std::string dtype_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
