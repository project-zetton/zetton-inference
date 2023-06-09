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

/// \brief InferenceRuntime class is used to load and infer model on
/// different devices
class InferenceRuntime {
 public:
  /// \brief default constructor
  InferenceRuntime() = default;

  /// \brief default destructor
  ~InferenceRuntime() = default;

 public:
  /// \brief initialize inference runtime with options
  /// \param options options for inference runtime
  /// \return true if success, otherwise false
  bool Init(const InferenceRuntimeOptions& options);

  /// \brief infer model with input tensors and return output tensors
  /// \param[in] input_tensors input tensors
  /// \param[out] output_tensors output tensors (inference results)
  /// \return true if success, otherwise false
  /// \n the name of input tensor should be the same as the name of input node
  /// in model
  bool Infer(std::vector<Tensor>& input_tensors,
             std::vector<Tensor>* output_tensors);

  /// \brief infer model with already binded input tensors and output tensors
  /// \return true if success, otherwise false
  /// \n the input tensors and output tensors should be binded before calling
  /// this function
  bool Infer();

 public:
  // Getters

  /// \brief get the number of inputs of inference backend
  /// \return the number of inputs
  int NumInputs();

  /// \brief get the number of outputs of inference backend
  /// \return the number of outputs
  int NumOutputs();

  /// \brief get the input info of inference backend
  /// \param index the index of input tensor
  /// \return the input tensor info
  TensorInfo GetInputInfo(int index);

  /// \brief get all of the input infos of inference backend
  /// \return the input tensor infos
  std::vector<TensorInfo> GetInputInfos();

  /// \brief get the output info of inference backend
  /// \param index the index of output tensor
  /// \return the output tensor info
  TensorInfo GetOutputInfo(int index);

  /// \brief get all of the output infos of inference backend
  /// \return the output tensor infos
  std::vector<TensorInfo> GetOutputInfos();

  /// \brief get the output tensor
  /// \param name the name of the output tensor
  /// \return the output tensor
  Tensor* GetOutputTensor(const std::string& name);

 public:
  // Setters

  /// \brief bind input tensor to inference backend
  /// \param name the name of the input tensor
  /// \param tensor the input tensor
  /// \details bind without copying and just use the data pointer of input
  /// tensor
  void BindInputTensor(const std::string& name, Tensor& tensor);

  /// \brief bind output tensor to inference backend
  /// \param name the name of the output tensor
  /// \param tensor the output tensor
  /// \details bind without copying and just use the data pointer of output
  /// tensor
  void BindOutputTensor(const std::string& name, Tensor& tensor);

 public:
  // Operations

  /// \brief clone inference runtime
  /// \param stream the CUDA stream of cloned inference runtime
  /// \param device_id the device id of cloned inference runtime
  /// \return the cloned inference runtime
  /// \n To avoid potential issues when multiple instances of the same model are
  /// created, it is recommended to clone a new runtime. This will ensure that
  /// each instance has its own runtime and will not interfere with the others.
  InferenceRuntime* Clone(void* stream = nullptr, int device_id = -1);

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
  InferenceRuntimeOptions options;
  /// \brief backend engine for model inference runtime
  std::unique_ptr<BaseInferenceBackend> backend_;
  /// \brief binded input tensors
  std::vector<Tensor> input_tensors_;
  /// \brief binded output tensors
  std::vector<Tensor> output_tensors_;
};

}  // namespace inference
}  // namespace zetton
