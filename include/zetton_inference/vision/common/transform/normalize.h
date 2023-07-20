#pragma once

#include "zetton_inference/vision/common/transform/base.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Normalize the input image by subtracting mean and dividing std
class Normalize : public BaseTransform {
 public:
  /// \brief constructor
  /// \param alpha scale factor
  /// \param beta offset
  Normalize(const std::vector<float>& alpha, const std::vector<float>& beta);

  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "Normalize"; }

  /// \brief transform the input image
  /// \details Compute result = mat * alpha + beta directly by channel.
  /// The default behavior is the same as OpencV's convertTo method.
  /// \param mat input image
  /// \param alpha scale factor
  /// \param beta offset
  /// \param lib transform library to be used
  static bool Run(Mat* mat, const std::vector<float>& alpha,
                  const std::vector<float>& beta,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  /// \brief scale factor
  std::vector<float> alpha_;
  /// \brief offset
  std::vector<float> beta_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
