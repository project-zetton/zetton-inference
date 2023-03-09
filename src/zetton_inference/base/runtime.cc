#include "zetton_inference/base/runtime.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/base/options.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_backend.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

bool InferenceRuntime::Init(InferenceRuntimeOptions* input_options) {
  options_ = input_options;

  // check inference frontend
  if (options_->model_format == InferenceFrontendType::kAuto) {
    options_->model_format = GuessModelFormat(options_->model_file);
    AINFO_F("Use guessed model format: {}", ToString(options_->model_format));
  }

  // check inference backend
  if (options_->backend == InferenceBackendType::kUnknown) {
    if (IsBackendAvailable(InferenceBackendType::kONNXRuntime)) {
      AINFO_F("Use ONNXRuntime backend by default.");
      options_->backend = InferenceBackendType::kONNXRuntime;
    } else if (IsBackendAvailable(InferenceBackendType::kOpenVINO)) {
      options_->backend = InferenceBackendType::kOpenVINO;
      AINFO_F("Use OpenVINO backend by default.");
    } else {
      AWARN_F("No available backend found.");
      return false;
    }
  }

  // check device for inference backend
  if (options_->backend == InferenceBackendType::kONNXRuntime) {
    // ONNX Runtime backend supports both CPU and GPU
    ACHECK_F(options_->device == InferenceDeviceType::kCPU ||
                 options_->device == InferenceDeviceType::kGPU,
             "{} only supports {} and {}.", ToString(options_->backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForONNXRuntime();
  } else if (options_->backend == InferenceBackendType::kTensorRT) {
    // TensorRT backend supports GPU only
    ACHECK_F(options_->device == InferenceDeviceType::kGPU,
             "{} only supports {}.", ToString(options_->backend),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForTensorRT();
  } else if (options_->backend == InferenceBackendType::kNCNN) {
    // NCNN backend supports both CPU and GPU
    ACHECK_F(options_->device == InferenceDeviceType::kCPU ||
                 options_->device == InferenceDeviceType::kGPU,
             "{} only supports {} and {}.", ToString(options_->backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kGPU));
    CreateBackendForNCNN();
  } else if (options_->backend == InferenceBackendType::kOpenVINO) {
    // OpenVINO backend supports CPU only
    ACHECK_F(options_->device == InferenceDeviceType::kCPU,
             "{} only supports {}.", ToString(options_->backend),
             ToString(InferenceDeviceType::kCPU));
    CreateBackendForOpenVINO();
  } else if (options_->backend == InferenceBackendType::kRKNN) {
    // RKNN backend supports both CPU and NPU
    ACHECK_F(options_->device == InferenceDeviceType::kCPU ||
                 options_->device == InferenceDeviceType::kNPU,
             "{} only supports {} and {}.", ToString(options_->backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kNPU));
    CreateBackendForRKNN();
  } else {
    AINFO_F("Invalid backend: {}.", ToString(options_->backend));
    return false;
  }

  AINFO_F("Runtime initialized with backend {} in device {}.",
          ToString(options_->backend), ToString(options_->device));

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
  ACHECK_F(backend_->Init(options_), "Failed to init TensorRT backend.");
}

}  // namespace inference
}  // namespace zetton
