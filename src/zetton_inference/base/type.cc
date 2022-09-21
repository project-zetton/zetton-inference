#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

std::string ToString(const InferenceDevice& device) {
  if (device == InferenceDevice::kCPU) {
    return "CPU";
  } else if (device == InferenceDevice::kGPU) {
    return "GPU";
  } else {
    return "Unknown";
  }
}

std::string ToString(const InferenceBackendType& backend) {
  if (backend == InferenceBackendType::kTensorRT) {
    return "TensorRT";
  } else if (backend == InferenceBackendType::kONNXRuntime) {
    return "ONNXRuntime";
  } else if (backend == InferenceBackendType::kNCNN) {
    return "NCNN";
  } else if (backend == InferenceBackendType::kOpenVINO) {
    return "OpenVINO";
  } else if (backend == InferenceBackendType::kRKNN) {
    return "RKNN";
  } else {
    return "Unknown";
  }
}

std::string ToString(const InferenceFrontendType& frontend) {
  if (frontend == InferenceFrontendType::kONNX) {
    return "ONNX";
  } else {
    return "Unknown";
  }
}

}  // namespace inference
}  // namespace zetton
