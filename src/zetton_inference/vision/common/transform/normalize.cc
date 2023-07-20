#include "zetton_inference/vision/common/transform/normalize.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/vision/common/transform/base.h"

namespace zetton {
namespace inference {
namespace vision {

Normalize::Normalize(const std::vector<float>& alpha,
                     const std::vector<float>& beta) {
  ACHECK_F(alpha.size() == beta.size(),
           "alpha and beta must have the same size");
  ACHECK_F(alpha.size() != 0, "alpha and beta must have at least one element");
  alpha_.assign(alpha.begin(), alpha.end());
  beta_.assign(beta.begin(), beta.end());
}

bool Normalize::RunOnOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::merge(split_im, *im);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Normalize::RunOnOpenCVCUDA(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  std::vector<cv::cuda::GpuMat> split_im;
  cv::cuda::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::cuda::merge(split_im, *im);
  return true;
}
#endif

bool Normalize::Run(Mat* mat, const std::vector<float>& alpha,
                    const std::vector<float>& beta, TransformLibraryType lib) {
  auto c = Normalize(alpha, beta);
  return c(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
