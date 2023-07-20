#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {

struct InferenceRuntimeOptions {
 public:
  // Constructor and destructor

  /// \brief default constructor
  InferenceRuntimeOptions() = default;

  /// \brief default destructor
  virtual ~InferenceRuntimeOptions() = default;

 public:
  // Setter

  /// \brief set path of model file and params file
  /// \param model_path path of model file
  /// \param params_path path of params file
  /// \param input_model_format format of input model file
  void SetModelPath(
      const std::string& model_path, const std::string& params_path = "",
      const InferenceFrontendType& format = InferenceFrontendType::kONNX);

  /// \brief set the buffer of model and params
  /// \param model_buffer buffer of model
  /// \param params_buffer buffer of params
  /// \param input_model_format format of input model buffer
  void SetModelBuffer(
      const std::string& model_buffer, const std::string& params_buffer = "",
      const InferenceFrontendType& format = InferenceFrontendType::kONNX);

  /// \brief set number of thread while inference in CPU
  /// \param thread_num number of thread
  void SetCpuThreadNum(int thread_num);

 public:
  // Operations

  /// \brief set model inference in GPU
  void UseCpu();
  /// \brief set model inference in CPU
  /// \param gpu_id id of GPU (or device)
  void UseGpu(int gpu_id = 0);

  /// \brief use ONNX Runtime backend
  void UseONNXRuntimeBackend();
  /// \brief use TensorRT backend
  void UseTensorRTBackend();
  /// \brief use NCNN backend
  void UseNCNNBackend();
  /// \brief use OpenVINO backend
  void UseOpenVINOBackend();
  /// \brief use RKNN backend
  void UseRKNNBackend();

 public:
  // Attributes

  /// \brief format of input original model
  InferenceFrontendType model_format = InferenceFrontendType::kAuto;

  /// \brief backend engine for model inference
  InferenceBackendType backend = InferenceBackendType::kUnknown;

  /// \brief device for model infernce
  InferenceDeviceType device = InferenceDeviceType::kCPU;

  /// \brief device id (e.g. GPU id) for model inference
  int device_id = 0;

  /// \brief for cpu inference and preprocess (-1 to let the backend choose
  /// their own default value)
  int cpu_thread_num = -1;

  /// \brief path of model file
  std::string model_file = "";
  /// \brief path of parameters file, can be empty
  std::string params_file = "";
  /// \brief the flag of model file is from memory buffer or not
  bool model_from_memory = false;
};

}  // namespace inference
}  // namespace zetton
