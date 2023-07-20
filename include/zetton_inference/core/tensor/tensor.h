#pragma once

#include <vector>

#include "zetton_inference/core/tensor/scalar.h"
#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {

/// \brief Tensor class is a representation of data matrix (or named tensor)
struct Tensor {
 public:
  // Constructors and Destructors

  /// \brief default constructor
  Tensor() = default;
  /// \brief constructor with tensor name
  /// \param name name of tensor
  explicit Tensor(const std::string& tensor_name);
  /// \brief constructor with tensor name
  /// \param name name of tensor
  explicit Tensor(const char* tensor_name);
  /// \brief constructor with a scalar
  /// \param scalar scalar value
  explicit Tensor(const Scalar& scalar);

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
  // Getters

  /// \brief get data buffer pointer
  void* MutableData();

  /// \brief get data buffer pointer
  void* Data();

  /// \brief get const data buffer pointer
  const void* Data() const;

  /// \brief get data buffer pointer in cpu
  /// \details Use this data to obtain the tensor data for processing. Since
  /// the most common scenario involves processing data on the CPU, this
  /// function will return a pointer to CPU memory buffer. If the original data
  /// is located on another device, the data will be copied to the CPU and
  /// stored in a temporary CPU buffer.
  const void* CpuData() const;

  /// \brief Whether the tensor owns the data buffer or shares it from outside.
  /// \return true if the tensor owns the data buffer, false if it shares the
  /// data buffer from outside.
  bool IsShared();

  /// \brief get total size of tensor memory buffer in bytes
  int Nbytes() const;

  /// \brief get total number of elements in this tensor
  int Numel() const;

  /// \brief print debug messages
  void Print(const std::string& prefix = "TensorInfo: ");

 public:
  // Setters

  /// \brief set data for tensor
  /// \param tensor_shape shape of tensor
  /// \param data_type data type of tensor
  /// \param data_buffer pointer to data buffer
  /// \param copy whether to copy data from data_buffer to internal buffer. If
  /// set to false, this tensor will not share memory with the data buffer and
  /// userself will be responsible for managing the data. Please review for any
  /// further errors.
  /// \param data_device device type of data buffer
  /// \param data_device_id device id of data buffer
  void SetData(
      const std::vector<int64_t>& tensor_shape,
      const InferenceDataType& data_type, void* data_buffer, bool copy = false,
      const InferenceDeviceType& data_device = InferenceDeviceType::kCPU,
      int data_device_id = -1);

  /// \brief set user memory buffer for tensor
  /// \details the external memory buffer is not managed by Tensor, so the user
  /// should take care of the memory buffer
  /// \param new_shape shape of tensor
  /// \param data_type data type of tensor
  /// \param data_buffer pointer to data buffer
  /// \param new_device device type of tensor
  /// \param new_device_id device id of tensor
  void SetExternalData(
      const std::vector<int64_t>& new_shape, const InferenceDataType& data_type,
      void* data_buffer,
      const InferenceDeviceType& new_device = InferenceDeviceType::kCPU,
      int new_device_id = -1);

 public:
  // Operations

  /// \brief Stop sharing memory with external data buffer
  /// \details If the tensor is sharing the data buffer with external pointer,
  /// this method will copy it to its own structure. Otherwise, it will do
  /// nothing.
  void StopSharing();

  /// \brief Reallocate the tensor memory buffer with new size in bytes
  void Allocate(const std::vector<int64_t>& new_shape,
                const InferenceDataType& data_type);

  /// \brief initialize Tensor
  /// \details set attribute for tensor and allocate cpu memory buffer
  /// \param new_shape shape of tensor
  /// \param data_type data type of tensor
  /// \param new_name name of tensor
  /// \param new_device device type of tensor
  void Allocate(const std::vector<int64_t>& new_shape,
                const InferenceDataType& data_type,
                const std::string& tensor_name,
                const InferenceDeviceType& new_device);

  /// \brief expand the shape of a Tensor
  /// \details insert a new axis that will appear at the `axis` position in the
  /// expanded Tensor shape.
  /// \param axis position where new axis is inserted
  /// \n It will not change the data memory, just modify its attribute `shape`
  void ExpandDim(int64_t axis = 0);

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

 private:
  // Internal helper functions

  /// \brief re-allocate memory buffer for tensor
  bool ReallocFn(size_t nbytes);

  /// \brief free memory buffer for tensor
  void FreeFn();

  /// \brief copy data from other tensor
  /// \param dst destination memory buffer
  /// \param src source memory buffer
  /// \param nbytes number of bytes to copy
  /// \param data_device device type of data
  /// \param data_is_pinned_memory whether the data buffer is in pinned memory
  void CopyBuffer(
      void* dst, const void* src, size_t nbytes,
      const InferenceDeviceType& data_device = InferenceDeviceType::kCPU,
      bool data_is_pinned_memory = false);

 public:
  // Attributes

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
  /// \details The internal data will be on the CPU. Sometimes, the external
  /// data may be on the GPU and we will use the GPU to infer the model. This
  /// enables us to skip data transfer, potentially improving efficiency.
  InferenceDeviceType device = InferenceDeviceType::kCPU;

  /// \brief tensor device id
  /// \n by default, device_id is -1, which means the device id is not set
  /// and will use the same device id as the inference runtime
  int device_id = -1;

  /// \brief whether the data buffer is in pinned memory, which is allocated
  /// with cudaMallocHost()
  bool is_pinned_memory = false;

  /// \brief temporary cpu memory buffer
  /// \details If the external data is not stored on the CPU, we utilize a
  /// temporary buffer to transfer the data to the CPU. There may be instances
  /// where we need to access data from other devices.
  /// \n this buffer will be allocated when the first time we need to
  /// access the data on other devices
  std::vector<int8_t> temporary_cpu_buffer;

  /// \brief The total number of bytes that have been allocated up to this
  /// point.
  /// \details During the process of resizing GPU memory, we will only free and
  /// reallocate memory if the new required size exceeds the value of this
  /// variable.
  size_t nbytes_allocated = 0;
};

}  // namespace inference
}  // namespace zetton
