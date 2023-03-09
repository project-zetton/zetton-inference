#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

/// \brief options for inference runtime
struct InferenceRuntimeOptions {
 public:
  InferenceRuntimeOptions() = default;
  virtual ~InferenceRuntimeOptions() = default;

 public:
  /// \brief set path of model file and params file
  /// \param model_path path of model file
  /// \param params_path path of params file
  /// \param input_model_format format of input model file
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const InferenceFrontendType& input_model_format =
                        InferenceFrontendType::kONNX);

  /// \brief set memory buffer of model file and params file
  /// \param model_buffer memory buffer of model file in string format
  /// \param params_buffer memory buffer of params file in string format
  /// \param input_model_format format of input model file
  void SetModelBuffer(const std::string& model_buffer,
                      const std::string& params_buffer = "",
                      const InferenceFrontendType& input_model_format =
                          InferenceFrontendType::kONNX);

  /// \brief set encryption key for model file
  /// \n encryption key is used to decrypt model file if it is encrypted
  /// \param encryption_key encryption key for model file
  void SetEncryptionKey(const std::string& input_encryption_key);

  /// \brief set model inference in CPU
  void UseCpu();

  /// \brief set model inference in GPU
  /// \param gpu_id id of GPU (or device)
  void UseGpu(int gpu_id = 0);

  /// \brief set number of thread while inference in CPU
  /// \param thread_num number of thread
  void SetCpuThreadNum(int thread_num);

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

  /// \brief Graph optimization level for ONNX Runtime model inference
  /// \details -1: use default value of ONNX Runtime
  /// 0: disable all optimizations
  /// 1: enable basic optimizations
  /// 2: enable extended optimizations
  /// 3: enable all optimizations
  int ort_graph_opt_level = -1;
  /// \brief Number of threads for ONNX Runtime model inference
  int ort_inter_op_num_threads = -1;
  /// \brief Execution mode for ONNX Runtime model inference
  /// \details 0: sequential
  /// 1: parallel
  int ort_execution_mode = -1;

  /// \brief path of model file
  std::string model_file = "";
  /// \brief path of parameters file, can be empty
  std::string params_file = "";
  /// \brief whether or not the model file is from memory buffer
  /// \details if true, the model file and params file is binary data in string
  bool model_from_memory = false;
  /// \brief encryption key for model file
  std::string encryption_key = "";
};

}  // namespace inference
}  // namespace zetton
