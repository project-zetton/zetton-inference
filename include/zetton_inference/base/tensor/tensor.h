#pragma once

#include <vector>

#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

struct Tensor {
 public:
  /// \brief default constructor
  Tensor() = default;
  /// \brief constructor with tensor name
  explicit Tensor(const std::string& tensor_name);

  /// \brief deep copy constructor
  Tensor(const Tensor& other);
  /// \brief move constructor
  Tensor(Tensor&& other) noexcept;

  /// \brief deep copy assignment
  Tensor& operator=(const Tensor& other);
  /// \brief move assignment
  Tensor& operator=(Tensor&& other) noexcept;

  /// \brief destructor
  ~Tensor() { FreeFn(); }

 public:
  /// \brief get data buffer pointer
  void* MutableData();

  /// \brief get data buffer pointer
  void* Data();

  /// \brief get const data buffer pointer
  const void* Data() const;

  /// \brief get data buffer pointer in cpu
  /// if the original data is on other device, the data
  /// will copy to cpu store in `temporary_cpu_buffer`
  const void* CpuData() const;

 public:
  /// \brief set user memory buffer for tensor
  /// the memory is managed by
  /// the user it self, but the Tensor will share the memory with user
  /// So take care with the user buffer
  void SetExternalData(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      void* data_buffer,
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  /// \brief expand the shape of a Tensor
  /// insert a new axis that will appear at the `axis` position in the expanded
  /// Tensor shape.
  void ExpandDim(int64_t axis = 0);

  /// \brief initialize Tensor
  // include setting attribute for tensor and allocate cpu memory buffer
  void Allocate(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  /// \brief get total size of tensor memory buffer in bytes
  int Nbytes() const;

  /// \brief get total number of elements in this tensor
  int Numel() const;

  /// \brief resize the tensor memory buffer with new size in bytes
  void Resize(size_t nbytes);

  /// \brief resize the tensor memory buffer with new shape
  void Resize(const std::vector<int64_t>& new_shape);

  /// \brief resize the tensor memory buffer with new shape, data type, name and
  /// device
  void Resize(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

 public:
  /// \brief print debug messages
  void PrintInfo(const std::string& prefix = "TensorInfo: ");

 private:
  /// \brief re-allocate memory buffer for tensor
  bool ReallocFn(size_t nbytes);

  /// \brief free memory buffer for tensor
  void FreeFn();

  /// \brief copy data from other tensor
  void CopyBuffer(void* dst, const void* src, size_t nbytes);

 public:
  /// \brief inner memory buffer for tensor
  void* buffer_ = nullptr;

  /// \brief user allocated memory buffer
  /// this use to skip memory copy step, the external_data_ptr will point to the
  /// user allocated memory user has to maintain the memory, allocate and
  /// release
  void* external_data_ptr = nullptr;

  /// \brief tensor data type
  InferenceDataType dtype = InferenceDataType::kINT8;

  /// \brief tensor shape
  std::vector<int64_t> shape = {0};

  /// \brief tensor name
  std::string name = "";

  // The internal data will be on CPU
  // Some times, the external data is on the GPU, and we are going to use
  // GPU to inference the model
  // so we can skip data transfer, which may improve the efficience
  InferenceDeviceType device = InferenceDeviceType::kCPU;

  /// \brief temporary cpu memory buffer
  // if the external data is not on CPU, we use this temporary buffer
  // to transfer data to CPU at some cases we need to visit the
  // other devices' data
  std::vector<int8_t> temporary_cpu_buffer;
};

}  // namespace inference
}  // namespace zetton
