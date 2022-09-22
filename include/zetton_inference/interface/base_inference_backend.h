#pragma once

#include <map>
#include <memory>
#include <vector>

#include "zetton_common/util/registerer.h"
#include "zetton_inference/base/tensor/tensor.h"
#include "zetton_inference/base/tensor/tensor_info.h"

namespace zetton {
namespace inference {

class BaseInferenceBackend {
 public:
  BaseInferenceBackend() = default;
  virtual ~BaseInferenceBackend() = default;

  virtual bool Initialized() const { return initialized_; }

  virtual int NumInputs() const = 0;
  virtual int NumOutputs() const = 0;
  virtual TensorInfo GetInputInfo(int index) = 0;
  virtual TensorInfo GetOutputInfo(int index) = 0;
  virtual bool Infer(std::vector<Tensor>& inputs,
                     std::vector<Tensor>* outputs) = 0;

 private:
  bool initialized_ = false;
};

ZETTON_REGISTER_REGISTERER(BaseInferenceBackend)
#define ZETTON_REGISTER_INFERENCE_BACKEND(name) \
  ZETTON_REGISTER_CLASS(BaseInferenceBackend, name)

}  // namespace inference
}  // namespace zetton
