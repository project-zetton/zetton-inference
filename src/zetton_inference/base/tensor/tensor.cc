#include "zetton_inference/base/tensor/tensor.h"

#include <absl/strings/str_join.h>

#include <cstdint>
#include <numeric>
#include <sstream>

#include "zetton_inference/base/common.h"
#include "zetton_inference/base/tensor/allocate.h"
#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

Tensor::Tensor(const std::string& tensor_name) { name = tensor_name; }

Tensor::Tensor(const Tensor& other)
    : external_data_ptr(other.external_data_ptr),
      dtype(other.dtype),
      shape(other.shape),
      name(other.name),
      device(other.device) {
  // Copy buffer
  if (other.buffer_ == nullptr) {
    buffer_ = nullptr;
  } else {
    size_t nbytes = Nbytes();
    ACHECK_F(ReallocFn(nbytes), "Failed to allocate memory for tensor.");
    CopyBuffer(buffer_, other.buffer_, nbytes);
  }
}

Tensor::Tensor(Tensor&& other) noexcept
    : buffer_(other.buffer_),
      external_data_ptr(other.external_data_ptr),
      dtype(other.dtype),
      shape(std::move(other.shape)),
      name(std::move(other.name)),
      device(other.device) {
  other.name = "";
  other.buffer_ = nullptr;
  other.external_data_ptr = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (&other != this) {
    // Copy buffer
    if (other.buffer_ == nullptr) {
      FreeFn();
      buffer_ = nullptr;
    } else {
      Resize(other.shape);
      size_t nbytes = Nbytes();
      CopyBuffer(buffer_, other.buffer_, nbytes);
    }

    shape = other.shape;
    name = other.name;
    dtype = other.dtype;
    device = other.device;
    external_data_ptr = other.external_data_ptr;
  }
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (&other != this) {
    FreeFn();
    buffer_ = other.buffer_;
    external_data_ptr = other.external_data_ptr;

    shape = std::move(other.shape);
    name = std::move(other.name);
    dtype = other.dtype;
    device = other.device;

    other.name = "";
    other.buffer_ = nullptr;
    other.external_data_ptr = nullptr;
  }
  return *this;
}

void* Tensor::MutableData() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

void* Tensor::Data() {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

const void* Tensor::Data() const {
  if (external_data_ptr != nullptr) {
    return external_data_ptr;
  }
  return buffer_;
}

const void* Tensor::CpuData() const {
  if (device == InferenceDeviceType::kGPU) {
#if USE_GPU == 1
    auto* cpu_ptr = const_cast<std::vector<int8_t>*>(&temporary_cpu_buffer);
    cpu_ptr->resize(Nbytes());
    // need to copy cuda mem to cpu first
    if (external_data_ptr != nullptr) {
      ACHECK_F(cudaMemcpy(cpu_ptr->data(), external_data_ptr, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "Failed to copy memory from GPU to CPU.");

    } else {
      ACHECK_F(cudaMemcpy(cpu_ptr->data(), buffer_, Nbytes(),
                          cudaMemcpyDeviceToHost) == 0,
               "Failed to copy memory from GPU to CPU.");
    }
    return cpu_ptr->data();
#else
    AFATAL_F(
        "Definition USE_GPU is not set and this operation may cause undefined "
        "behavior.");
#endif
  }
  return Data();
}

void Tensor::SetExternalData(const std::vector<int64_t>& new_shape,
                             const InferenceDataType& data_type,
                             void* data_buffer,
                             const InferenceDeviceType& new_device) {
  dtype = data_type;
  shape.assign(new_shape.begin(), new_shape.end());
  external_data_ptr = data_buffer;
  device = new_device;
}

void Tensor::ExpandDim(int64_t axis) {
  size_t ndim = shape.size();
  ACHECK_F(axis >= 0 && axis <= static_cast<int64_t>(ndim),
           "Invalid axis {} for ndim: {}.", axis, ndim);
  shape.insert(shape.begin() + axis, 1);
}

void Tensor::Allocate(const std::vector<int64_t>& new_shape,
                      const InferenceDataType& data_type,
                      const std::string& tensor_name,
                      const InferenceDeviceType& new_device) {
  dtype = data_type;
  name = tensor_name;
  shape.assign(new_shape.begin(), new_shape.end());
  device = new_device;
  size_t nbytes = Nbytes();
  ACHECK_F(ReallocFn(nbytes), "Failed to allocate memory for tensor.");
}

int Tensor::Nbytes() const { return Numel() * InferenceDataTypeSize(dtype); }

int Tensor::Numel() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void Tensor::Resize(size_t new_nbytes) { ReallocFn(new_nbytes); }

void Tensor::Resize(const std::vector<int64_t>& new_shape) {
  int numel = Numel();
  int new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                  std::multiplies<int>());
  if (new_numel > numel) {
    size_t nbytes = static_cast<long>(new_numel) * InferenceDataTypeSize(dtype);
    ReallocFn(nbytes);
  }
  shape.assign(new_shape.begin(), new_shape.end());
}

void Tensor::Resize(const std::vector<int64_t>& new_shape,
                    const InferenceDataType& data_type,
                    const std::string& tensor_name,
                    const InferenceDeviceType& new_device) {
  name = tensor_name;
  device = new_device;
  dtype = data_type;
  int new_nbytes = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>()) *
                   InferenceDataTypeSize(data_type);
  ReallocFn(new_nbytes);
  shape.assign(new_shape.begin(), new_shape.end());
}

template <typename T>
void CalculateTensorStats(void* src_ptr, int size, double* mean, double* max,
                          double* min) {
  T* ptr = static_cast<T*>(src_ptr);
  *mean = 0;
  *max = -99999999;
  *min = 99999999;
  for (int i = 0; i < size; ++i) {
    if (*(ptr + i) > *max) {
      *max = *(ptr + i);
    }
    if (*(ptr + i) < *min) {
      *min = *(ptr + i);
    }
    *mean += *(ptr + i);
  }
  *mean = *mean / size;
}

void Tensor::Print(const std::string& prefix) {
  double mean = 0;
  double max = -99999999;
  double min = 99999999;
  if (dtype == InferenceDataType::kFP32) {
    CalculateTensorStats<float>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == InferenceDataType::kFP64) {
    CalculateTensorStats<double>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == InferenceDataType::kINT8) {
    CalculateTensorStats<int8_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == InferenceDataType::kUINT8) {
    CalculateTensorStats<uint8_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == InferenceDataType::kINT32) {
    CalculateTensorStats<int32_t>(Data(), Numel(), &mean, &max, &min);
  } else if (dtype == InferenceDataType::kINT64) {
    CalculateTensorStats<int64_t>(Data(), Numel(), &mean, &max, &min);
  } else {
    AFATAL_F("Unsupported data type: {}.", ToString(dtype));
  }
  AINFO_F("{}: name={}, shape={}, dtype={}, mean={}, max={}, min={}", prefix,
          name, absl::StrJoin(shape, " "), ToString(dtype), mean, max, min);
}

bool Tensor::ReallocFn(size_t nbytes) {
  if (device == InferenceDeviceType::kGPU) {
#if USE_GPU == 1
    size_t original_nbytes = Nbytes();
    if (nbytes > original_nbytes) {
      if (buffer_ != nullptr) {
        ZettonFreeDevice(buffer_);
      }
      ZettonMallocDevice(&buffer_, nbytes);
    }
    return buffer_ != nullptr;
#else
    AFATAL_F(
        "Definition USE_GPU is not set and this operation may cause "
        "undefined behavior.");
#endif
  }
  buffer_ = realloc(buffer_, nbytes);
  return buffer_ != nullptr;
}

void Tensor::FreeFn() {
  if (external_data_ptr != nullptr) external_data_ptr = nullptr;
  if (buffer_ != nullptr) {
    if (device == InferenceDeviceType::kGPU) {
#if USE_GPU == 1
      ZettonFreeDevice(buffer_);
#endif
    } else {
      ZettonFreeHost(buffer_, false);
    }
    buffer_ = nullptr;
  }
}

void Tensor::CopyBuffer(void* dst, const void* src, size_t nbytes) {
  if (device == InferenceDeviceType::kGPU) {
#if USE_GPU == 1
    ACHECK_F(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice) == 0,
             "Failed to copy data from device to device.");

#else
    AFATAL_F(
        "Definition USE_GPU is not set and this operation may cause "
        "undefined behavior.");
#endif
  } else {
    std::memcpy(dst, src, nbytes);
  }
}

}  // namespace inference
}  // namespace zetton
