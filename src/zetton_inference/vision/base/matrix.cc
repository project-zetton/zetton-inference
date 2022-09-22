#include "zetton_inference/vision/base/matrix.h"

#include <absl/strings/str_join.h>

#include "zetton_common/log/log.h"
#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {
namespace vision {

#ifdef ENABLE_OPENCV_CUDA
cv::cuda::GpuMat* Mat::GetGpuMat() {
  if (device == InferenceDeviceType::kCPU) {
    gpu_mat.upload(cpu_mat);
  }
  return &gpu_mat;
}
#endif

cv::Mat* Mat::GetCpuMat() {
#ifdef ENABLE_OPENCV_CUDA
  if (device == InferenceDeviceType::kGPU) {
    gpu_mat.download(cpu_mat);
  }
#endif
  return &cpu_mat;
}

void Mat::ShareWithTensor(Tensor* tensor) {
  if (device == InferenceDeviceType::kGPU) {
#ifdef ENABLE_OPENCV_CUDA
    tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                            GetGpuMat()->ptr());
    tensor->device = InferenceDeviceType::kGPU;
#endif
  } else {
    tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                            GetCpuMat()->ptr());
    tensor->device = InferenceDeviceType::kCPU;
  }
  if (layout == TensorLayoutType::kHWC) {
    tensor->shape = {Height(), Width(), Channels()};
  }
}

bool Mat::CopyToTensor(Tensor* tensor) {
  cv::Mat* im = GetCpuMat();
  int total_bytes = im->total() * im->elemSize();
  if (total_bytes != tensor->Nbytes()) {
    AERROR_F(
        "While copy Mat to Tensor, requires the memory size be same, "
        "but now size of Tensor = {}, size of Mat = {}.",
        tensor->Nbytes(), total_bytes);
    return false;
  }
  memcpy(tensor->MutableData(), im->ptr(), im->total() * im->elemSize());
  return true;
}

void Mat::PrintInfo(const std::string& flag) {
  cv::Mat* im = GetCpuMat();
  cv::Scalar mean = cv::mean(*im);
  std::vector<double> mean_vec;
  for (int i = 0; i < Channels(); ++i) {
    mean_vec.push_back(mean[i]);
  }
  AINFO_F("{}: Channel={}, height={}, width={}, mean={}", flag, Channels(),
          Height(), Width(), absl::StrJoin(mean_vec, " "));
}

InferenceDataType Mat::Type() {
  int type = -1;
  if (device == InferenceDeviceType::kGPU) {
#ifdef ENABLE_OPENCV_CUDA
    type = gpu_mat.type();
#endif
  } else {
    type = cpu_mat.type();
  }
  if (type < 0) {
    AFATAL_F(
        "While calling Mat::Type(), get negative value, which is not "
        "expected!.");
  }
  type = type % 8;
  if (type == 0) {
    return InferenceDataType::kUINT8;
  } else if (type == 1) {
    return InferenceDataType::kINT8;
  } else if (type == 2) {
    AFATAL_F(
        "While calling Mat::Type(), get UINT16 type which is not "
        "supported now.");
  } else if (type == 3) {
    return InferenceDataType::kINT16;
  } else if (type == 4) {
    return InferenceDataType::kINT32;
  } else if (type == 5) {
    return InferenceDataType::kFP32;
  } else if (type == 6) {
    return InferenceDataType::kFP64;
  } else {
    AFATAL_F(
        "While calling Mat::Type(), get type = {}, which is not expected!.",
        type);
  }

  return InferenceDataType::kUnknwon;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
