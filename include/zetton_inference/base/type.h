#pragma once

#include <string>

namespace zetton {
namespace inference {

enum class InferenceDevice { kUnknwon = 0, kCPU, kGPU };

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

std::string ToString(const InferenceDevice& device);
std::string ToString(const InferenceBackendType& backend);
std::string ToString(const InferenceFrontendType& frontend);

}  // namespace inference
}  // namespace zetton
