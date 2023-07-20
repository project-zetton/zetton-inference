#pragma once

#include <map>
#include <string>
#include <vector>

namespace zetton {
namespace inference {

/// \brief device type for model inference
enum class InferenceDeviceType { kUnknwon = 0, kCPU, kGPU, kRKNPU2 };

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
  kAuto,
  kONNX,
  kSerialized,
  kRKNN,
  // kPyTorch,
  // kTensorFlow,
};

/// \brief all supported backend types w.r.t. frontend type (model format)
static std::map<InferenceFrontendType, std::vector<InferenceBackendType>>
    s_default_backends_by_format = {
        {
            InferenceFrontendType::kONNX,
            {
                InferenceBackendType::kTensorRT,
                InferenceBackendType::kONNXRuntime,
                InferenceBackendType::kNCNN,
                InferenceBackendType::kOpenVINO,
            },
        },
        {
            InferenceFrontendType::kSerialized,
            {
                InferenceBackendType::kTensorRT,
            },
        },
        {
            InferenceFrontendType::kRKNN,
            {
                InferenceBackendType::kRKNN,
            },
        },
};

/// \brief all supported backend types w.r.t. device type
static std::map<InferenceDeviceType, std::vector<InferenceBackendType>>
    s_default_backends_by_device = {
        {
            InferenceDeviceType::kCPU,
            {
                InferenceBackendType::kONNXRuntime,
                InferenceBackendType::kOpenVINO,
                InferenceBackendType::kNCNN,
            },
        },
        {
            InferenceDeviceType::kGPU,
            {
                InferenceBackendType::kTensorRT,
                InferenceBackendType::kONNXRuntime,
                InferenceBackendType::kNCNN,
            },
        },
        {
            InferenceDeviceType::kRKNPU2,
            {
                InferenceBackendType::kRKNN,
            },
        },
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
