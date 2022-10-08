#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Add padding to the input image
class Pad : public BaseTransform {
 public:
  /// \brief constructor
  /// \param top top padding
  /// \param bottom bottom padding
  /// \param left left padding
  /// \param right right padding
  /// \param value padding values
  Pad(int top, int bottom, int left, int right,
      const std::vector<float>& value) {
    top_ = top;
    bottom_ = bottom;
    left_ = left;
    right_ = right;
    value_ = value;
  }

  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpenCVCUDA(Mat* mat);
#endif

  /// \brief name of the transform
  std::string Name() override { return "Pad"; }

  /// \brief transform the input image
  /// \param mat input image
  /// \param top top padding
  /// \param bottom bottom padding
  /// \param left left padding
  /// \param right right padding
  /// \param value padding values
  /// \param lib transform library to be used
  static bool Run(Mat* mat, const int& top, const int& bottom, const int& left,
                  const int& right, const std::vector<float>& value,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  /// \brief top padding
  int top_;
  /// \brief bottom padding
  int bottom_;
  /// \brief left padding
  int left_;
  /// \brief right padding
  int right_;
  /// \brief padding values
  std::vector<float> value_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
