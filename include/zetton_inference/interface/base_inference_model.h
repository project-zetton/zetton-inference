#pragma once

#include <string>

#include "zetton_inference/base/options.h"
#include "zetton_inference/base/runtime.h"

namespace zetton {
namespace inference {

class BaseInferenceModel {
 public:
  virtual bool Init(const InferenceRuntimeOptions& options) = 0;
  virtual bool Infer(std::vector<Tensor>& input_tensors,
                     std::vector<Tensor>* output_tensors);
  virtual std::string Name() const { return "BaseInferenceModel"; }

 public:
  virtual bool InitRuntime();
  virtual bool CreateCpuBackend();
  virtual bool CreateGpuBackend();

 public:
  virtual int NumInputsOfRuntime() { return runtime_->NumInputs(); }
  virtual int NumOutputsOfRuntime() { return runtime_->NumOutputs(); }
  virtual TensorInfo InputInfoOfRuntime(int index) {
    return runtime_->GetInputInfo(index);
  }
  virtual TensorInfo OutputInfoOfRuntime(int index) {
    return runtime_->GetOutputInfo(index);
  }
  virtual bool Initialized() const {
    return runtime_initialized_ && initialized;
  }

 public:
  InferenceRuntimeOptions runtime_options;
  std::vector<InferenceBackendType> valid_cpu_backends = {
      InferenceBackendType::kONNXRuntime};
  std::vector<InferenceBackendType> valid_gpu_backends = {
      InferenceBackendType::kONNXRuntime};
  std::vector<InferenceBackendType> valid_external_backends;
  bool initialized = false;

 private:
  std::unique_ptr<InferenceRuntime> runtime_;
  bool runtime_initialized_ = false;
  bool debug_ = false;
};

}  // namespace inference
}  // namespace zetton
