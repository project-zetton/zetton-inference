#pragma once

#include <opencv2/core/core.hpp>
#ifdef ENABLE_OPENCV_CUDA
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#endif

#include "zetton_inference/core/tensor/tensor.h"
#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {
namespace vision {

enum TensorLayoutType { kHWC, kCHW };

std::string ToString(TensorLayoutType type);

/// \brief Matrix is a wrapper of Tensor, which is used to store image data
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
  /// \brief get cv::cuda::GpuMat on GPU
  cv::cuda::GpuMat* GetGpuMat();
#endif
  /// \brief get cv::Mat on CPU
  cv::Mat* GetCpuMat();

  /// \brief get data type of the matrix
  InferenceDataType Type();
  /// \brief get the number of channels of the matrix
  int Channels() const { return channels; }
  /// \brief get the width of the matrix
  int Width() const { return width; }
  /// \brief get the height of the matrix
  int Height() const { return height; }
  /// \brief set the number of channels of the matrix
  void SetChannels(int s) { channels = s; }
  /// \brief set the width of the matrix
  void SetWidth(int w) { width = w; }
  /// \brief set the height of the matrix
  void SetHeight(int h) { height = h; }

  /// \brief use external data to initialize the matrix
  void ShareWithTensor(Tensor* tensor);
  /// \brief copy the matrix to the tensor
  bool CopyToTensor(Tensor* tensor);

  /// \brief print debug information
  void Print(const std::string& flag);

 public:
  /// \brief the layout of the matrix
  TensorLayoutType layout = TensorLayoutType::kHWC;
  /// \brief the device of the matrix
  InferenceDeviceType device = InferenceDeviceType::kCPU;

 private:
  /// \brief the number of channels of the matrix
  int channels;
  /// \brief the width of the matrix
  int height;
  /// \brief the height of the matrix
  int width;
  /// \brief the cv::Mat on CPU
  cv::Mat cpu_mat;
#ifdef ENABLE_OPENCV_CUDA
  /// \brief the cv::cuda::GpuMat on GPU
  cv::cuda::GpuMat gpu_mat;
#endif
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
