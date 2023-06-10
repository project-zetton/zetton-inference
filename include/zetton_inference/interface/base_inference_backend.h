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
  /// \brief get information of all the input tensors
  virtual std::vector<TensorInfo> GetInputInfos() {
    AERROR_F("GetInputInfos is not supported.");
    return {};
  }
  /// \brief get information of all the output tensors
  virtual std::vector<TensorInfo> GetOutputInfos() {
    AERROR_F("GetOutputInfos is not supported.");
    return {};
  }

 public:
  /// \brief initialize inference backend with options
  /// \param options options for inference backend
  /// \return true if success, otherwise false
  virtual bool Init(const InferenceRuntimeOptions* options) = 0;

  /// \brief run model inference with input tensors and save the results to
  /// output tensors
  /// \param input_tensors input tensors
  /// \param output_tensors output tensors
  virtual bool Infer(std::vector<Tensor>& inputs,
                     std::vector<Tensor>* outputs) = 0;

  // Optional: For those backends which can share memory
  // while creating multiple inference engines with same model file
  virtual std::unique_ptr<BaseInferenceBackend> Clone(
      const InferenceRuntimeOptions* runtime_option, void* stream = nullptr,
      int device_id = -1) {
    AERROR_F("Cloning is not supported for backend [{}] on device [{}]",
             ToString(runtime_option->backend),
             ToString(runtime_option->device));
    return nullptr;
  }

 protected:
  /// \brief whether the backend is initialized
  bool initialized_ = false;
};

ZETTON_REGISTER_REGISTERER(BaseInferenceBackend)
#define ZETTON_REGISTER_INFERENCE_BACKEND(name) \
  ZETTON_REGISTER_CLASS(BaseInferenceBackend, name)

}  // namespace inference
}  // namespace zetton
