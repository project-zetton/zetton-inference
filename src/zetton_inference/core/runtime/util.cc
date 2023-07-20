#include "zetton_inference/core/runtime/util.h"

#include <algorithm>

#include "zetton_common/log/log.h"
#include "zetton_inference/backend/base.h"
#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {

std::vector<InferenceBackendType> GetAvailableBackends() {
  // init available backends
  std::vector<InferenceBackendType> backends;

  // check backend availabilities in backend object factory
  if (BaseInferenceBackendRegisterer::IsValid("TensorRTInferenceBackend")) {
    backends.push_back(InferenceBackendType::kTensorRT);
  }
  if (BaseInferenceBackendRegisterer::IsValid("ONNXRuntimeInferenceBackend")) {
    backends.push_back(InferenceBackendType::kONNXRuntime);
  }
  if (BaseInferenceBackendRegisterer::IsValid("NCNNInferenceBackend")) {
    backends.push_back(InferenceBackendType::kNCNN);
  }
  if (BaseInferenceBackendRegisterer::IsValid("OpenVINOInferenceBackend")) {
    backends.push_back(InferenceBackendType::kOpenVINO);
  }
  if (BaseInferenceBackendRegisterer::IsValid("RKNNInferenceBackend")) {
    backends.push_back(InferenceBackendType::kRKNN);
  }

  return backends;
}

bool IsBackendAvailable(const InferenceBackendType& backend) {
  // get all available backends
  std::vector<InferenceBackendType> backends = GetAvailableBackends();
  // check if the specified backend is available
  return std::find_if(backends.begin(), backends.end(),
                      [&](auto& b) { return b == backend; }) != backends.end();
}

bool CheckModelFormat(const std::string& model_file,
                      const InferenceFrontendType& model_format) {
  if (model_format == InferenceFrontendType::kSerialized) {
    //
  } else if (model_format == InferenceFrontendType::kONNX) {
    if (model_file.size() < 5 ||
        model_file.substr(model_file.size() - 5, 5) != ".onnx") {
      AERROR_F(
          "The model file is not a ONNX file since the file name does not "
          "end with '.onnx': {}",
          model_file);
      return false;
    }
  } else {
    AERROR_F("Unsupported model format: {}", ToString(model_format));
    return false;
  }
  return true;
}

InferenceFrontendType GuessModelFormat(const std::string& model_file) {
  if (model_file.size() > 5 &&
      model_file.substr(model_file.size() - 5, 5) == ".onnx") {
    AINFO_F("The model file is a ONNX file: {}", model_file);
    return InferenceFrontendType::kONNX;
  }

  AERROR_F("Cannot guess the model format of the model file: {}", model_file);
  return InferenceFrontendType::kUnknown;
}

bool IsBackendSupported(const InferenceFrontendType& model_format,
                        const InferenceBackendType& backend) {
  // get supported backends for the specified model format
  auto iter = s_default_backends_by_format.find(model_format);
  if (iter == s_default_backends_by_format.end()) {
    AERROR_F("Cannot find the default backend for the model format: {}",
             ToString(model_format));
    return false;
  }
  // check if the specified backend is supported
  for (auto& i : iter->second) {
    if (i == backend) {
      // the specified backend is supported
      return true;
    }
  }
  // the specified backend is not supported
  AERROR_F(
      "The specified backend is not supported by the specified model "
      "format: {} vs. {}",
      ToString(backend), ToString(model_format));
  return false;
}

bool IsBackendSupported(const InferenceDeviceType& device,
                        const InferenceBackendType& backend) {
  // get supported backends for the specified device
  auto iter = s_default_backends_by_device.find(device);
  if (iter == s_default_backends_by_device.end()) {
    AERROR_F("Cannot find the default backend for the device: {}",
             ToString(device));
    return false;
  }
  // check if the specified backend is supported
  for (auto& i : iter->second) {
    if (i == backend) {
      // the specified backend is supported
      return true;
    }
  }
  // the specified backend is not supported
  AERROR_F(
      "The specified backend is not supported by the specified device: {} "
      "vs. {}",
      ToString(backend), ToString(device));
  return false;
}

bool AutoSelecteBackend(const InferenceFrontendType& model_format,
                        const InferenceDeviceType& device,
                        InferenceBackendType& backend) {
  // find the default backend for the specified model format
  auto backends_supported_by_format =
      s_default_backends_by_format.find(model_format);
  if (backends_supported_by_format == s_default_backends_by_format.end()) {
    AERROR_F("Cannot found a default backend for model format: {}",
             ToString(model_format));
    return false;
  }

  // find the default backend for the specified device
  auto backends_supported_by_device = s_default_backends_by_device.find(device);
  if (backends_supported_by_device == s_default_backends_by_device.end()) {
    AERROR_F("Cannot found a default backend for device: {}", ToString(device));
    return false;
  }

  // find the intersection of the two sets
  std::vector<InferenceBackendType> backends;
  for (auto& i : backends_supported_by_format->second) {
    for (auto& j : backends_supported_by_device->second) {
      if (i == j) {
        backends.push_back(i);
      }
    }
  }

  // check if the intersection is empty
  if (backends.empty()) {
    AERROR_F(
        "Cannot found a default backend for the specified model format and "
        "device: {} vs. {}",
        ToString(model_format), ToString(device));
    return false;
  }

  // check if the backend is available
  for (auto& i : backends) {
    if (IsBackendAvailable(i)) {
      backend = i;
      AINFO_F("Auto selected backend: {}", ToString(backend));
      return true;
    }
  }

  // the intersection is not empty, but no backend is available
  AERROR_F(
      "Cannot found an available backend for the specified model format and "
      "device: {} vs. {}",
      ToString(model_format), ToString(device));
  return false;
}

}  // namespace inference
}  // namespace zetton
