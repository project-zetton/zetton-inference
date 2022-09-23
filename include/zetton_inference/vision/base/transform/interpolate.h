#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

class Interpolate : public BaseTransform {
 public:
  Interpolate(int width, int height, float scale_w = -1.0, float scale_h = -1.0,
              int interp = 1, bool use_scale = false) {
    width_ = width;
    height_ = height;
    scale_w_ = scale_w;
    scale_h_ = scale_h;
    interp_ = interp;
    use_scale_ = use_scale;
  }

  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpneCVCUDA(Mat* mat) override;
#endif
  std::string Name() override { return "Resize"; }

  static bool Run(Mat* mat, int width, int height, float scale_w = -1.0,
                  float scale_h = -1.0, int interp = 1, bool use_scale = false,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

  bool SetWidthAndHeight(int width, int height) {
    width_ = width;
    height_ = height;
    return true;
  }

  std::tuple<int, int> GetWidthAndHeight() {
    return std::make_tuple(width_, height_);
  }

 private:
  int width_;
  int height_;
  float scale_w_ = -1.0;
  float scale_h_ = -1.0;
  int interp_ = 1;
  bool use_scale_ = false;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
