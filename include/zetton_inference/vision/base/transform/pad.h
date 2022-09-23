#pragma once

#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

class Pad : public BaseTransform {
 public:
  Pad(int top, int bottom, int left, int right,
      const std::vector<float>& value) {
    top_ = top;
    bottom_ = bottom;
    left_ = left;
    right_ = right;
    value_ = value;
  }
  bool RunOnOpenCV(Mat* mat) override;
#ifdef ENABLE_OPENCV_CUDA
  bool RunOnOpenCVCUDA(Mat* mat);
#endif
  std::string Name() override { return "Pad"; }

  static bool Run(Mat* mat, const int& top, const int& bottom, const int& left,
                  const int& right, const std::vector<float>& value,
                  TransformLibraryType lib = TransformLibraryType::kOpenCV);

 private:
  int top_;
  int bottom_;
  int left_;
  int right_;
  std::vector<float> value_;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
