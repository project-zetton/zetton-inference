#pragma once

#include <opencv2/core/core.hpp>
#ifdef ENABLE_OPENCV_CUDA
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#endif

#include "zetton_inference/base/tensor/tensor.h"
#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {
namespace vision {

enum TensorLayoutType { kHWC, kCHW };

struct Mat {
  explicit Mat(cv::Mat& mat) {
    cpu_mat = mat;
    device = InferenceDeviceType::kCPU;
    layout = TensorLayoutType::kHWC;
    height = cpu_mat.rows;
    width = cpu_mat.cols;
    channels = cpu_mat.channels();
  }

 public:
#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat* GetGpuMat();
#endif
  cv::Mat* GetCpuMat();

  InferenceDataType Type();
  int Channels() const { return channels; }
  int Width() const { return width; }
  int Height() const { return height; }
  void SetChannels(int s) { channels = s; }
  void SetWidth(int w) { width = w; }
  void SetHeight(int h) { height = h; }

  // Transfer the vision::Mat to FDTensor
  void ShareWithTensor(Tensor* tensor);
  // Only support copy to cpu tensor now
  bool CopyToTensor(Tensor* tensor);

  // debug functions
  // This function will print shape / mean of each channels of the Mat
  void PrintInfo(const std::string& flag);

 public:
  TensorLayoutType layout = TensorLayoutType::kHWC;
  InferenceDeviceType device = InferenceDeviceType::kCPU;

 private:
  int channels;
  int height;
  int width;
  cv::Mat cpu_mat;
#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat gpu_mat;
#endif
};

}  // namespace vision

}  // namespace inference
}  // namespace zetton
