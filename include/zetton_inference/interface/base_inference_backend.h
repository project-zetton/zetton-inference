#pragma once

#include <map>
#include <memory>
#include <vector>

#include "zetton_common/util/registerer.h"
#include "zetton_inference/base/options.h"
#include "zetton_inference/base/tensor/tensor.h"
#include "zetton_inference/base/tensor/tensor_info.h"

namespace zetton {
namespace inference {

class BaseInferenceBackend {
 public:
  /// \brief InferenceBackend constructor
  BaseInferenceBackend() = default;
  /// \brief InferenceBackend destructor
  virtual ~BaseInferenceBackend() = default;

 public:
  /// \brief whether the backend is initialized
  virtual bool Initialized() const { return initialized_; }
  /// \brief get the number of inputs
  virtual int NumInputs() const = 0;
  /// \brief get the number of outputs
  virtual int NumOutputs() const = 0;
  /// \brief get the input tensor info
  /// \param index input tensor index
  virtual TensorInfo GetInputInfo(int index) = 0;
  /// \brief get the output tensor info
  /// \param index index of output tensor
  virtual TensorInfo GetOutputInfo(int index) = 0;

 public:
  virtual bool Init(const InferenceRuntimeOptions& options) = 0;
  /// \brief run model inference with input tensors and save the results to
  /// output tensors
  /// \param input_tensors input tensors
  /// \param output_tensors output tensors
  virtual bool Infer(std::vector<Tensor>& inputs,
                     std::vector<Tensor>* outputs) = 0;

 protected:
  /// \brief whether the backend is initialized
  bool initialized_ = false;
};

ZETTON_REGISTER_REGISTERER(BaseInferenceBackend)
#define ZETTON_REGISTER_INFERENCE_BACKEND(name) \
  ZETTON_REGISTER_CLASS(BaseInferenceBackend, name)

}  // namespace inference
}  // namespace zetton
