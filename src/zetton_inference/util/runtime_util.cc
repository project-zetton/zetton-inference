#include "zetton_inference/util/runtime_util.h"

#include <algorithm>

#include "zetton_common/log/log.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_backend.h"

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
          "With model format of {}, the model file "
          "should ends with `.onnx`, but now it's {}",
          ToString(model_format), model_file);
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
    AINFO_F("Model Format: ONNX.");
    return InferenceFrontendType::kONNX;
  }

  AERROR_F(
      "Cannot guess which model format you are using, please set manually.");
  return InferenceFrontendType::kUnknown;
}

}  // namespace inference
}  // namespace zetton
