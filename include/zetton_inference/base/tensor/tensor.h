#pragma once

#include <vector>

#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

struct Tensor {
 public:
  Tensor() = default;
  explicit Tensor(const std::string& tensor_name);

  // Deep copy
  Tensor(const Tensor& other);
  // Move constructor
  Tensor(Tensor&& other) noexcept ;

  // Deep copy assignment
  Tensor& operator=(const Tensor& other);
  // Move assignment
  Tensor& operator=(Tensor&& other) noexcept ;

  ~Tensor() { FreeFn(); }

 public:
  // Get data buffer pointer
  void* MutableData();

  void* Data();

  const void* Data() const;

  // Use this data to get the tensor data to process
  // Since the most senario is process data in CPU
  // this function will return a pointer to cpu memory
  // buffer.
  // If the original data is on other device, the data
  // will copy to cpu store in `temporary_cpu_buffer`
  const void* CpuData() const;

 public:
  // Set user memory buffer for Tensor, the memory is managed by
  // the user it self, but the Tensor will share the memory with user
  // So take care with the user buffer
  void SetExternalData(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      void* data_buffer,
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  // Expand the shape of a Tensor. Insert a new axis that will appear
  // at the `axis` position in the expanded Tensor shape.
  void ExpandDim(int64_t axis = 0);

  // Initialize Tensor
  // Include setting attribute for tensor
  // and allocate cpu memory buffer
  void Allocate(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  // Total size of tensor memory buffer in bytes
  int Nbytes() const;

  // Total number of elements in this tensor
  int Numel() const;

  void Resize(size_t nbytes);

  void Resize(const std::vector<int64_t>& new_shape);

  void Resize(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  // Debug function
  // Use this function to print shape, dtype, mean, max, min
  // prefix will also be printed as tag
  void PrintInfo(const std::string& prefix = "TensorInfo: ");

  bool ReallocFn(size_t nbytes);

  void FreeFn();

 private:
  void CopyBuffer(void* dst, const void* src, size_t nbytes);

 public:
  // std::vector<int8_t> data;
  void* buffer_ = nullptr;
  std::vector<int64_t> shape = {0};
  std::string name = "";
  InferenceDataType dtype = InferenceDataType::kINT8;

  // This use to skip memory copy step
  // the external_data_ptr will point to the user allocated memory
  // user has to maintain the memory, allocate and release
  void* external_data_ptr = nullptr;

  // The internal data will be on CPU
  // Some times, the external data is on the GPU, and we are going to use
  // GPU to inference the model
  // so we can skip data transfer, which may improve the efficience
  InferenceDeviceType device = InferenceDeviceType::kCPU;

  // if the external data is not on CPU, we use this temporary buffer
  // to transfer data to CPU at some cases we need to visit the
  // other devices' data
  std::vector<int8_t> temporary_cpu_buffer;
};

}  // namespace inference
}  // namespace zetton
