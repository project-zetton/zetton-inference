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

  // choose backend automatically if not specified
  if (options.backend == InferenceBackendType::kUnknown) {
    if (!AutoSelecteBackend(options.model_format, options.device,
                            options.backend)) {
      AWARN_F("Failed to select backend automatically.");
      return false;
    }
  }

  // create inference backend
  if (options.backend == InferenceBackendType::kONNXRuntime) {
    CreateBackendForONNXRuntime();
  } else if (options.backend == InferenceBackendType::kTensorRT) {
    CreateBackendForTensorRT();
  } else if (options.backend == InferenceBackendType::kNCNN) {
    CreateBackendForNCNN();
  } else if (options.backend == InferenceBackendType::kOpenVINO) {
    CreateBackendForOpenVINO();
  } else if (options.backend == InferenceBackendType::kRKNN) {
    CreateBackendForRKNN();
  } else {
    AINFO_F("Invalid backend: {}.", ToString(options.backend));
    return false;
  }

  AINFO_F(
      "Runtime initialized with backend [{}] with model format [{}] on device "
      "{}.",
      ToString(options.backend), ToString(options.model_format),
      ToString(options.device));

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
  AINFO_F("Runtime Clone with Backend:: {} in {}.", options.backend,
          options.device);
  runtime->options = options;
  runtime->backend_ = backend_->Clone(options, stream, device_id);
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
  ACHECK_F(backend_->Init(options), "Failed to init TensorRT backend.");
}

}  // namespace inference
}  // namespace zetton
