#include "zetton_inference/base/options.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

void InferenceRuntimeOptions::SetModelPath(
    const std::string& model_path, const std::string& params_path,
    const InferenceFrontendType& format) {
  model_file = model_path;
  params_file = params_path;
  model_format = format;
  model_from_memory = false;
}

void InferenceRuntimeOptions::SetModelBuffer(
    const std::string& model_buffer, const std::string& params_buffer,
    const InferenceFrontendType& format) {
  model_file = model_buffer;
  params_file = params_buffer;
  model_format = format;
  model_from_memory = true;
}

void InferenceRuntimeOptions::UseGpu(int gpu_id) {
#ifdef USE_GPU
  device = InferenceDeviceType::kGPU;
  device_id = gpu_id;
#else
  AWARN_F("Definition USE_GPU is not set, use CPU forcely.")
  device = InferenceDeviceType::kCPU;
#endif
}

void InferenceRuntimeOptions::UseCpu() { device = InferenceDeviceType::kCPU; }

void InferenceRuntimeOptions::SetCpuThreadNum(int thread_num) {
  ACHECK_F(thread_num > 0, "Invalid thread number: {}", thread_num);
  cpu_thread_num = thread_num;
}

void InferenceRuntimeOptions::UseONNXRuntimeBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kONNXRuntime),
           "ONNXRuntime backend is not available.");
  backend = InferenceBackendType::kONNXRuntime;
}

void InferenceRuntimeOptions::UseTensorRTBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kTensorRT),
           "TensorRT backend is not available.");
  backend = InferenceBackendType::kTensorRT;
}

void InferenceRuntimeOptions::UseNCNNBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kNCNN),
           "NCNN backend is not available.");
  backend = InferenceBackendType::kNCNN;
}

void InferenceRuntimeOptions::UseOpenVINOBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kOpenVINO),
           "OpenVINO backend is not available.");
  backend = InferenceBackendType::kOpenVINO;
}

void InferenceRuntimeOptions::UseRKNNBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kRKNN),
           "RKNN backend is not available.");
  backend = InferenceBackendType::kRKNN;
}

void InferenceRuntimeOptions::SetInputShapeForTensorRT(
    const std::string& input_name, const std::vector<int32_t>& min_shape,
    const std::vector<int32_t>& opt_shape,
    const std::vector<int32_t>& max_shape) {
  trt_min_shape[input_name].clear();
  trt_max_shape[input_name].clear();
  trt_opt_shape[input_name].clear();
  trt_min_shape[input_name].assign(min_shape.begin(), min_shape.end());
  if (opt_shape.size() == 0) {
    trt_opt_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_opt_shape[input_name].assign(opt_shape.begin(), opt_shape.end());
  }
  if (max_shape.size() == 0) {
    trt_max_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_max_shape[input_name].assign(max_shape.begin(), max_shape.end());
  }
}

void InferenceRuntimeOptions::EnableFP16ForTensorRT() {
  trt_enable_fp16 = true;
}

void InferenceRuntimeOptions::DisableFP16ForTensorRT() {
  trt_enable_fp16 = false;
}

void InferenceRuntimeOptions::SetCacheFileForTensorRT(
    const std::string& cache_file_path) {
  trt_serialize_file = cache_file_path;
}

}  // namespace inference
}  // namespace zetton
