#pragma once

#include "zetton_inference/vision/common/transform/base.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief Resize the input image
class Interpolate : public BaseTransform {
 public:
  /// \brief constructor of Interpolate
  /// \param width width of the output image
  /// \param height height of the output image
  /// \param scale_w scale factor of width
  /// \param scale_h scale factor of height
  /// \param interp interpolation method
  /// \param use_scale whether to use scale factor
  Interpolate(int width, int height, float scale_w = -1.0, float scale_h = -1.0,
              int interp = 1, bool use_scale = false) {
    width_ = width;
    height_ = height;
    scale_w_ = scale_w;
    scale_h_ = scale_h;
    interp_ = interp;
    use_scale_ = use_scale;
  }

  /// \brief transform the input image on CPU
  bool RunOnOpenCV(Mat* mat) override;

#ifdef ENABLE_OPENCV_CUDA
  /// \brief transform the input image on GPU
  bool RunOnOpneCVCUDA(Mat* mat) override;
#endif

  /// \brief name of the transform
  std::string Name() override { return "Resize"; }

  /// \brief transform the input image
  /// \param mat input image
  /// \param width width of the output image
  /// \param height height of the output image
  /// \param scale_w scale factor of width
  /// \param scale_h scale factor of height
  /// \param interp interpolation method
  /// \param use_scale whether to use scale factor
  /// \param lib transform library to be used
  static bool Run(Mat* mat, int width, int height, float scale_w = -1.0,
                  float scale_h = -1.0, int interp = 1, bool use_scale = false,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

  /// \brief set the desired size of the output image
  /// \param width desired width of the output image
  /// \param height desired height of the output image
  bool SetWidthAndHeight(int width, int height) {
    width_ = width;
    height_ = height;
    return true;
  }

  /// \brief get current width of the output image
  /// \param width current width of the output image
  /// \param height current height of the output image
  std::tuple<int, int> GetWidthAndHeight() {
    return std::make_tuple(width_, height_);
  }

 private:
  /// \brief width of the output image
  int width_;
  /// \brief height of the output image
  int height_;
  /// \brief scale factor of width
  float scale_w_ = -1.0;
  /// \brief scale factor of height
  float scale_h_ = -1.0;
  /// \brief interpolation method
  int interp_ = 1;
  /// \brief whether to use scale factor
  bool use_scale_ = false;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
