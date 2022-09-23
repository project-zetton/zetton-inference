#include "zetton_inference/vision/base/transform/permute.h"

#include <zetton_common/log/log.h>

#include "zetton_inference/base/type.h"
#include "zetton_inference/vision/base/matrix.h"
#include "zetton_inference/vision/interface/base_transform.h"

namespace zetton {
namespace inference {
namespace vision {

bool HWC2CHW::RunOnOpenCV(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("HWC2CHW: The input data is not HWC format!");
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  cv::Mat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();

  //  float* data = reinterpret_cast<float*>(im->data);
  for (int i = 0; i < rc; ++i) {
    //    cv::extractChannel(im_clone, cv::Mat(rh, rw, im->type() % 8, data + i
    //    * rh * rw),
    //                       i);
    cv::extractChannel(
        im_clone,
        cv::Mat(rh, rw, im->type() % 8,
                im->ptr() + i * rh * rw * InferenceDataTypeSize(mat->Type())),
        i);
  }
  mat->layout = TensorLayoutType::kCHW;
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool HWC2CHW::RunOnOpenCVCUDA(Mat* mat) {
  if (mat->layout != TensorLayoutType::kHWC) {
    AERROR_F("HWC2CHW: The input data is not HWC format!");
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::cuda::GpuMat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  int num_pixels = rh * rw;
  std::vector<cv::cuda::GpuMat> channels{
      cv::cuda::GpuMat(rh, rw, im->type() % 8, &(im->ptr()[0])),
      cv::cuda::GpuMat(rh, rw, im->type() % 8, &(im->ptr()[num_pixels])),
      cv::cuda::GpuMat(rh, rw, im->type() % 8, &(im->ptr()[num_pixels * 2]))};
  cv::cuda::split(im_clone, channels);
  mat->layout = TensorLayoutType::kCHW;
  return true;
}
#endif

bool HWC2CHW::Run(Mat* mat, TransformLibraryType lib) {
  auto h = HWC2CHW();
  return h(mat, lib);
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
