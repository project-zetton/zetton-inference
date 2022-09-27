#pragma once

#include <string>

namespace zetton {
namespace inference {

/// \brief device type for model inference
enum class InferenceDeviceType { kUnknwon = 0, kCPU, kGPU, kNPU };

/// \brief data type for model inference
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

/// \brief backend framework for model inference
enum class InferenceBackendType {
  kUnknown,
  kTensorRT,
  kONNXRuntime,
  kNCNN,
  kOpenVINO,
  kRKNN,
};

/// \brief frontend type (input model format) for model inference
enum class InferenceFrontendType {
  kUnknown,
  kSerialized,
  kAuto,
  kONNX,
  // kPyTorch,
  // kTensorFlow,
};

/// \brief convert InferenceDeviceType to std::string
std::string ToString(const InferenceDeviceType& device);
/// \brief conevrt InferenceDataType to std::string
std::string ToString(const InferenceDataType& dtype);
/// \brief convert InferenceBackendType to std::string
std::string ToString(const InferenceBackendType& backend);
/// \brief convert InferenceFrontendType to std::string
std::string ToString(const InferenceFrontendType& frontend);

/// \brief get byte size of InferenceDataType
int32_t InferenceDataTypeSize(const InferenceDataType& dtype);

}  // namespace inference
}  // namespace zetton
