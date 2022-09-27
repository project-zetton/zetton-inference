#include "zetton_inference/base/runtime.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/base/options.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_backend.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

bool InferenceRuntime::Init(const InferenceRuntimeOptions& input_options) {
  options = input_options;

  // check inference frontend
  if (options.model_format == InferenceFrontendType::kAuto) {
    options.model_format = GuessModelFormat(options.model_file);
    AINFO_F("Use guessed model format: {}", ToString(options.model_format));
  }

  // check inference backend
  if (options.backend == InferenceBackendType::kUnknown) {
    if (IsBackendAvailable(InferenceBackendType::kONNXRuntime)) {
      AINFO_F("Use ONNXRuntime backend by default.");
      options.backend = InferenceBackendType::kONNXRuntime;
    } else if (IsBackendAvailable(InferenceBackendType::kOpenVINO)) {
      options.backend = InferenceBackendType::kOpenVINO;
      AINFO_F("Use OpenVINO backend by default.");
    } else {
      AWARN_F("No available backend found.");
      return false;
    }
  }

  // check device for inference backend
  if (options.backend == InferenceBackendType::kONNXRuntime) {
    // ONNX Runtime backend supports both CPU and GPU
    ACHECK_F(options.device == InferenceDeviceType::kCPU ||
                 options.device == InferenceDeviceType::kGPU,
             "{} only supports {} and {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForONNXRuntime();
  } else if (options.backend == InferenceBackendType::kTensorRT) {
    // TensorRT backend supports GPU only
    ACHECK_F(options.device == InferenceDeviceType::kGPU,
             "{} only supports {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForTensorRT();
  } else if (options.backend == InferenceBackendType::kNCNN) {
    // NCNN backend supports both CPU and GPU
    ACHECK_F(options.device == InferenceDeviceType::kCPU ||
                 options.device == InferenceDeviceType::kGPU,
             "{} only supports {} and {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForNCNN();
  } else if (options.backend == InferenceBackendType::kOpenVINO) {
    // OpenVINO backend supports CPU only
    ACHECK_F(options.device == InferenceDeviceType::kCPU,
             "{} only supports {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU));
    CreateBackendForOpenVINO();
  } else if (options.backend == InferenceBackendType::kRKNN) {
    // RKNN backend supports both CPU and NPU
    ACHECK_F(options.device == InferenceDeviceType::kCPU ||
                 options.device == InferenceDeviceType::kNPU,
             "{} only supports {} and {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kNPU));
    CreateBackendForRKNN();
  } else {
    AINFO_F("Invalid backend: {}.", ToString(options.backend));
    return false;
  }

  AINFO_F("Runtime initialized with backend {} in device {}.",
          ToString(options.backend), ToString(options.device));

  return true;
}

void InferenceRuntime::CreateBackendForOpenVINO() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kOpenVINO),
           "OpenVINO backend is not available.");
  AFATAL_F("OpenVINO backend is not supported yet.");
}

void InferenceRuntime::CreateBackendForONNXRuntime() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kONNXRuntime),
           "ONNXRuntime backend is not available.");
  AFATAL_F("ONNXRuntime backend is not supported yet.");
}

void InferenceRuntime::CreateBackendForNCNN() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kNCNN),
           "NCNN backend is not available.");
  AFATAL_F("NCNN backend is not supported yet.");
}

void InferenceRuntime::CreateBackendForRKNN() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kRKNN),
           "RKNN backend is not available.");
  AFATAL_F("RKNN backend is not supported yet.");
}

void InferenceRuntime::CreateBackendForTensorRT() {
  // check if TensorRT backend is available
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kTensorRT),
           "TensorRT backend is not available.");
  // create TensorRT backend
  backend_.reset(BaseInferenceBackendRegisterer::GetInstanceByName(
      "TensorRTInferenceBackend"));
  // check if backend is created successfully
  ACHECK_F(backend_ != nullptr, "Failed to create TensorRT backend.");
  // initialize backend and check if it's successful
  ACHECK_F(backend_->Init(options), "Failed to init TensorRT backend.");
}

}  // namespace inference
}  // namespace zetton
