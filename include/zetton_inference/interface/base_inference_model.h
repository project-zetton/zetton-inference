#pragma once

#include <string>

#include "zetton_inference/base/options.h"
#include "zetton_inference/base/runtime.h"
#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

class BaseInferenceModel {
 public:
  /// \brief initialize inference model with options
  virtual bool Init(const InferenceRuntimeOptions& options) = 0;

  /// \brief run model inference with input tensors and save the results to
  /// output tensors
  /// \param input_tensors input tensors
  /// \param output_tensors output tensors
  virtual bool Infer(std::vector<Tensor>& input_tensors,
                     std::vector<Tensor>* output_tensors);

  /// \brief the name of the model
  virtual std::string Name() const { return "BaseInferenceModel"; }

 public:
  /// \brief initialize inference runtime
  virtual bool InitRuntime();
  /// \brief create inference backend on CPU
  virtual bool CreateCpuBackend();
  /// \brief create inference backend on GPU (or other devices)
  virtual bool CreateGpuBackend();
  /// \brief create inference backend on NPU (or other devices)
  virtual bool CreateNpuBackend();

 public:
  /// \brief get the number of inputs
  virtual int NumInputsOfRuntime() { return runtime_->NumInputs(); }
  /// \brief get the number of outputs
  virtual int NumOutputsOfRuntime() { return runtime_->NumOutputs(); }
  /// \brief get the input tensor info
  virtual TensorInfo InputInfoOfRuntime(int index) {
    return runtime_->GetInputInfo(index);
  }
  /// \brief get the output tensor info
  virtual TensorInfo OutputInfoOfRuntime(int index) {
    return runtime_->GetOutputInfo(index);
  }
  /// \brief get the initialized status of the inference model
  virtual bool Initialized() const {
    return runtime_initialized_ && initialized;
  }

 public:
  /// \brief start to record the inference time for statistics
  virtual void EnableRecordTimeOfRuntime() {
    time_of_runtime_.clear();
    std::vector<double>().swap(time_of_runtime_);
    enable_record_time_of_runtime_ = true;
  }

  /// \brief stop to record the inference time for statistics
  virtual void DisableRecordTimeOfRuntime() {
    enable_record_time_of_runtime_ = false;
  }

  /// \brief get the statistics of the inference time
  virtual std::map<std::string, double> PrintStatsInfoOfRuntime();

 public:
  /// \brief options of the inference model
  InferenceRuntimeOptions runtime_options;
  /// \brief available inference backends on CPU
  std::vector<InferenceBackendType> valid_cpu_backends = {
      InferenceBackendType::kONNXRuntime};
  /// \brief available inference backends on GPU (or other devices)
  std::vector<InferenceBackendType> valid_gpu_backends = {
      InferenceBackendType::kONNXRuntime};
  /// \brief available inference backends on GPU (or other devices)
  std::vector<InferenceBackendType> valid_npu_backends;
  /// \brief available external inference backends
  std::vector<InferenceBackendType> valid_external_backends;
  /// \brief whether the inference model is initialized
  bool initialized = false;

 private:
  /// \brief the runtime of the inference model
  std::unique_ptr<InferenceRuntime> runtime_;
  /// \brief whether the inference runtime is initialized
  bool runtime_initialized_ = false;
  /// \brief debug mode
  bool debug_ = false;

  /// \brief whether to record inference time
  bool enable_record_time_of_runtime_ = false;
  /// \brief record inference time for backend
  std::vector<double> time_of_runtime_;
};

}  // namespace inference
}  // namespace zetton
