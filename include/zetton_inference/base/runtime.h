#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "zetton_inference/base/options.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_backend.h"

namespace zetton {
namespace inference {

class InferenceRuntime {
 public:
  InferenceRuntime() = default;
  ~InferenceRuntime() = default;

 public:
  bool Init(const InferenceRuntimeOptions& options);
  bool Infer(std::vector<Tensor>& input_tensors,
             std::vector<Tensor>* output_tensors) {
    return backend_->Infer(input_tensors, output_tensors);
  };

 public:
  int NumInputs() { return backend_->NumInputs(); }
  int NumOutputs() { return backend_->NumOutputs(); }
  TensorInfo GetInputInfo(int index) { return backend_->GetInputInfo(index); }
  TensorInfo GetOutputInfo(int index) { return backend_->GetOutputInfo(index); }

 private:
  void CreateONNXRuntimeBackend();
  void CreateTensorRTBackend();
  void CreateOpenVINOBackend();

 private:
  /// \brief options for model inference runtime
  InferenceRuntimeOptions options;
  /// \brief backend engine for model inference runtime
  std::unique_ptr<BaseInferenceBackend> backend_;
};

}  // namespace inference
}  // namespace zetton
