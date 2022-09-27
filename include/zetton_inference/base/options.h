#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

class InferenceRuntimeOptions {
 public:
  InferenceRuntimeOptions() = default;
  ~InferenceRuntimeOptions() = default;

 public:
  /// \brief set path of model file and params file
  /// \param model_path path of model file
  /// \param params_path path of params file
  /// \param input_model_format format of input model file
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const std::string& input_model_format = "onnx");

  /// \brief set model inference in GPU
  void UseCpu();
  /// \brief set model inference in CPU
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

  /// \brief set input shape for TensorRT backend if the given model contains
  /// dynamic shape
  /// \details if opt_shape or max_shape are empty, they will keep same with the
  /// min_shape, which means the shape will be fixed as min_shape while
  /// inference
  /// \param input_name name of input tensor
  /// \param min_shape the minimum shape
  /// \param opt_shape the most common shape while inference, default be empty
  /// \param max_shape the maximum shape, default be empty
  void SetInputShapeForTensorRT(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<int32_t>());
  /// \brief enable half precision (FP16) for TensorRT backend
  void EnableFP16ForTensorRT();
  /// \brief disable half precision (FP16) and change to full precision (FP32)
  /// for TensorRT backend
  void DisableFP16ForTensorRT();
  /// \brief set path of cache file while using TensorRT backend
  /// \param cache_path path of cache file (serialized engine)
  void SetCacheFileForTensorRT(const std::string& cache_file_path);

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

  /// \brief maximum input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> trt_max_shape;
  /// \brief minimum input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> trt_min_shape;
  /// \brief optimal input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> trt_opt_shape;
  /// \brief serialized TensorRT model file
  std::string trt_serialize_file = "";
  /// \brief whether or not to enable FP16 precision in TensorRT model inference
  bool trt_enable_fp16 = false;
  /// \brief whether or not to enable INT8 precision in TensorRT model inference
  bool trt_enable_int8 = false;
  /// \brief maximum batch size for TensorRT model inference
  size_t trt_max_batch_size = 32;
  /// \brief maximum workspace size for TensorRT model inference
  size_t trt_max_workspace_size = 1 << 30;

  /// \brief path of model file
  std::string model_file = "";
  /// \brief path of parameters file, can be empty
  std::string params_file = "";
};

}  // namespace inference
}  // namespace zetton
