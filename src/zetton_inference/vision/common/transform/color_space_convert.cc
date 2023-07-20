#include "zetton_inference/vision/common/transform/color_space_convert.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef ENABLE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace zetton {
namespace inference {
namespace vision {

bool BGR2RGB::RunOnOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  cv::cvtColor(*im, *im, cv::COLOR_BGR2RGB);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool BGR2RGB::RunOnOpenCVCUDA(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::cuda::cvtColor(*im, *im, cv::COLOR_BGR2RGB);
  return true;
}
#endif

bool RGB2BGR::RunOnOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  cv::cvtColor(*im, *im, cv::COLOR_RGB2BGR);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool RGB2BGR::RunOnOpenCVCUDA(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::cuda::cvtColor(*im, *im, cv::COLOR_RGB2BGR);
  return true;
}
#endif

bool BGR2RGB::Run(Mat* mat, TransformLibraryType lib) {
  auto b = BGR2RGB();
  return b(mat, lib);
}

bool RGB2BGR::Run(Mat* mat, TransformLibraryType lib) {
  auto r = RGB2BGR();
  return r(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
