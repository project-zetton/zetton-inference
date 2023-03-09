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
  /// \brief initialize inference runtime with options
  /// \param options options for inference runtime
  bool Init(InferenceRuntimeOptions* options);

  /// \brief infer model with input tensors and return output tensors
  /// \param input_tensors input tensors
  /// \param output_tensors output tensors
  bool Infer(std::vector<Tensor>& input_tensors,
             std::vector<Tensor>* output_tensors) {
    return backend_->Infer(input_tensors, output_tensors);
  };

 public:
  /// \brief get number of inputs of inference backend
  int NumInputs() { return backend_->NumInputs(); }
  /// \brief get number of outputs of inference backend
  int NumOutputs() { return backend_->NumOutputs(); }
  /// \brief get input info of inference backend
  TensorInfo GetInputInfo(int index) { return backend_->GetInputInfo(index); }
  /// \brief get output info of inference backend
  TensorInfo GetOutputInfo(int index) { return backend_->GetOutputInfo(index); }

 private:
  /// \brief create ONNX Runtime backend
  void CreateBackendForONNXRuntime();
  /// \brief create TensorRT backend
  void CreateBackendForTensorRT();
  /// \brief create NCNN backend
  void CreateBackendForNCNN();
  /// \brief create OpenVINO backend
  void CreateBackendForOpenVINO();
  /// \brief create RKNN backend
  void CreateBackendForRKNN();

 private:
  /// \brief options for model inference runtime
  InferenceRuntimeOptions* options_;
  /// \brief backend engine for model inference runtime
  std::unique_ptr<BaseInferenceBackend> backend_;
};

}  // namespace inference
}  // namespace zetton
