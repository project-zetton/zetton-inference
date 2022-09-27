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
  /// \param name name of tensor
  explicit Tensor(const std::string& tensor_name);

  /// \brief deep copy constructor
  /// \param tensor tensor to be copied
  Tensor(const Tensor& other);
  /// \brief move constructor
  /// \param tensor tensor to be moved
  Tensor(Tensor&& other) noexcept;

  /// \brief deep copy assignment
  /// \param tensor tensor to be copied
  Tensor& operator=(const Tensor& other);
  /// \brief move assignment
  /// \param tensor tensor to be moved
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
  /// \details if tensor is not in cpu, it will be copied to cpu
  const void* CpuData() const;

 public:
  /// \brief set user memory buffer for tensor
  /// \details the external memory buffer is not managed by Tensor, so the user
  /// should take care of the memory buffer
  void SetExternalData(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      void* data_buffer,
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  /// \brief expand the shape of a Tensor
  /// \details insert a new axis that will appear at the `axis` position in the
  /// expanded Tensor shape.
  /// \param axis position where new axis is inserted
  void ExpandDim(int64_t axis = 0);

  /// \brief initialize Tensor
  /// \details set attribute for tensor and allocate cpu memory buffer
  /// \param new_shape shape of tensor
  /// \param data_type data type of tensor
  /// \param new_name name of tensor
  /// \param new_device device type of tensor
  void Allocate(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

  /// \brief get total size of tensor memory buffer in bytes
  int Nbytes() const;

  /// \brief get total number of elements in this tensor
  int Numel() const;

  /// \brief resize the tensor memory buffer with new size in bytes
  /// \param nbytes new size of tensor memory buffer in bytes
  void Resize(size_t nbytes);

  /// \brief resize the tensor memory buffer with new shape
  /// \param new_shape new shape of tensor
  void Resize(const std::vector<int64_t>& new_shape);

  /// \brief resize the tensor memory buffer with new shape, data type, name and
  /// device
  /// \param new_shape new shape of tensor
  /// \param data_type new data type of tensor
  /// \param tensor_name new name of tensor
  /// \param new_device new device of tensor
  void Resize(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      const std::string& tensor_name = "",
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU);

 public:
  /// \brief print debug messages
  void Print(const std::string& prefix = "TensorInfo: ");

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
  /// \details this is used for the case that the user want to use the external
  /// memory buffer to store the tensor data (to avoid the copy of data)
  void* external_data_ptr = nullptr;

  /// \brief tensor data type
  InferenceDataType dtype = InferenceDataType::kINT8;

  /// \brief tensor shape
  std::vector<int64_t> shape = {0};

  /// \brief tensor name
  std::string name = "";

  /// \brief tensor device type
  InferenceDeviceType device = InferenceDeviceType::kCPU;

  /// \brief temporary cpu memory buffer
  /// \details this buffer will be allocated when the first time we need to
  /// access the data on other devices
  std::vector<int8_t> temporary_cpu_buffer;
};

}  // namespace inference
}  // namespace zetton
