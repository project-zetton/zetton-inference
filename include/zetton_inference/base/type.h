#pragma once

#include <string>

namespace zetton {
namespace inference {

enum class InferenceDeviceType { kUnknwon = 0, kCPU, kGPU };

enum class InferenceDataType {
  kUnknwon = 0,
  kBOOL,
  kUINT8,
  kINT8,
  kINT16,
  kINT32,
  kINT64,
  kFP16,
  kFP32,
  kFP64,
};

enum class InferenceBackendType {
  kUnknown,
  kTensorRT,
  kONNXRuntime,
  kNCNN,
  kOpenVINO,
  kRKNN,
};

enum class InferenceFrontendType {
  kUnknown,
  kAuto,
  kONNX,
  // kPyTorch,
  // kTensorFlow,
};

std::string ToString(const InferenceDeviceType& device);
std::string ToString(const InferenceDataType& dtype);
std::string ToString(const InferenceBackendType& backend);
std::string ToString(const InferenceFrontendType& frontend);
int32_t InferenceDataTypeSize(const InferenceDataType& dtype);

}  // namespace inference
}  // namespace zetton
