#include "zetton_inference/core/runtime/runtime.h"

#include <algorithm>

#include "zetton_common/log/log.h"
#include "zetton_inference/backend/base.h"
#include "zetton_inference/core/runtime/options.h"
#include "zetton_inference/core/runtime/util.h"
#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {

bool InferenceRuntime::Init(InferenceRuntimeOptions* input_options) {
  options_ = input_options;

  // check inference frontend
  if (options_->model_format == InferenceFrontendType::kAuto) {
    options_->model_format = GuessModelFormat(options_->model_file);
    AINFO_F("Use guessed model format: {}", ToString(options_->model_format));
  }

  // choose backend automatically if not specified
  if (options_->backend == InferenceBackendType::kUnknown) {
    if (!AutoSelecteBackend(options_->model_format, options_->device,
                            options_->backend)) {
      AWARN_F("Failed to select backend automatically.");
      return false;
    }
  }

  // create inference backend
  if (options_->backend == InferenceBackendType::kONNXRuntime) {
    CreateBackendForONNXRuntime();
  } else if (options_->backend == InferenceBackendType::kTensorRT) {
    CreateBackendForTensorRT();
  } else if (options_->backend == InferenceBackendType::kNCNN) {
    CreateBackendForNCNN();
  } else if (options_->backend == InferenceBackendType::kOpenVINO) {
    CreateBackendForOpenVINO();
  } else if (options_->backend == InferenceBackendType::kRKNN) {
    CreateBackendForRKNN();
  } else {
    AINFO_F("Invalid backend: {}.", ToString(options_->backend));
    return false;
  }

  AINFO_F(
      "Runtime initialized with backend [{}] with model format [{}] on device "
      "{}.",
      ToString(options_->backend), ToString(options_->model_format),
      ToString(options_->device));

  return true;
}

bool InferenceRuntime::Infer(std::vector<Tensor>& input_tensors,
                             std::vector<Tensor>* output_tensors) {
  return backend_->Infer(input_tensors, output_tensors);
}

bool InferenceRuntime::Infer() { return true; }

int InferenceRuntime::NumInputs() { return backend_->NumInputs(); }

int InferenceRuntime::NumOutputs() { return backend_->NumOutputs(); }

TensorInfo InferenceRuntime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

std::vector<TensorInfo> InferenceRuntime::GetInputInfos() {
  return backend_->GetInputInfos();
}

TensorInfo InferenceRuntime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

std::vector<TensorInfo> InferenceRuntime::GetOutputInfos() {
  return backend_->GetOutputInfos();
}

Tensor* InferenceRuntime::GetOutputTensor(const std::string& name) {
  for (auto& output_tensor : output_tensors_) {
    if (output_tensor.name == name) {
      return &output_tensor;
    }
  }
  AWARN_F("Output tensor [{}] not found.", name);
  return nullptr;
}

void InferenceRuntime::BindInputTensor(const std::string& name,
                                       Tensor& tensor) {
  // try to find the tensor with the same name
  auto it = std::find_if(
      input_tensors_.begin(), input_tensors_.end(),
      [&name](const Tensor& tensor) { return tensor.name == name; });
  // if found, just set the external data
  if (it != input_tensors_.end()) {
    it->SetExternalData(tensor.shape, tensor.dtype, tensor.MutableData(),
                        tensor.device, tensor.device_id);
  }
  // if not, create a new tensor with external data
  else {
    Tensor new_tensor;
    new_tensor.SetExternalData(tensor.shape, tensor.dtype, tensor.MutableData(),
                               tensor.device, tensor.device_id);
    input_tensors_.emplace_back(std::move(new_tensor));
  }
}

void InferenceRuntime::BindOutputTensor(const std::string& name,
                                        Tensor& tensor) {
  // try to find the tensor with the same name
  auto it = std::find_if(
      output_tensors_.begin(), output_tensors_.end(),
      [&name](const Tensor& tensor) { return tensor.name == name; });
  // if found, just set the external data
  if (it != output_tensors_.end()) {
    it->SetExternalData(tensor.shape, tensor.dtype, tensor.MutableData(),
                        tensor.device, tensor.device_id);
  }
  // if not, create a new tensor with external data
  else {
    Tensor new_tensor;
    new_tensor.SetExternalData(tensor.shape, tensor.dtype, tensor.MutableData(),
                               tensor.device, tensor.device_id);
    output_tensors_.emplace_back(std::move(new_tensor));
  }
}

InferenceRuntime* InferenceRuntime::Clone(void* stream, int device_id) {
  auto* runtime = new InferenceRuntime();
  AINFO_F("Clone Runtime with backend [{}] on device [{}].",
          ToString(options_->backend), ToString(options_->device));
  runtime->options_ = options_;
  runtime->backend_ = backend_->Clone(options_, stream, device_id);
  return runtime;
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