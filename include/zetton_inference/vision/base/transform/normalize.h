#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

class Normalize : public BaseTransform {
 public:
  Normalize(const std::vector<float>& alpha, const std::vector<float>& beta);

  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif
  std::string Name() override { return "Normalize"; }

  // Compute `result = mat * alpha + beta` directly by channel.
  // The default behavior is the same as OpenCV's convertTo method.
  static bool Run(Mat* mat, const std::vector<float>& alpha,
                  const std::vector<float>& beta,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
