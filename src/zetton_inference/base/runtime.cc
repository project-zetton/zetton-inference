#include "zetton_inference/base/runtime.h"

#include <zetton_common/log/log.h>

#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

void InferenceRuntimeOptions::SetModelPath(const std::string& model_path,
                                           const std::string& params_path,
                                           const std::string& _model_format) {
  if (_model_format == "onnx") {
    model_file = model_path;
    model_format = InferenceFrontendType::kONNX;
  } else {
    AFATAL_F("The model format only can be 'onnx'.");
  }
}

void InferenceRuntimeOptions::UseGpu(int gpu_id) {
#ifdef USE_GPU
  device = InferenceDeviceType::kGPU;
  device_id = gpu_id;
#else
  AWARN_F("This project hasn't been compiled with GPU, use CPU forcely.");
  device = InferenceDeviceType::kCPU;
#endif
}

void InferenceRuntimeOptions::UseCpu() { device = InferenceDeviceType::kCPU; }

void InferenceRuntimeOptions::SetCpuThreadNum(int thread_num) {
  ACHECK_F(thread_num > 0, "The thread_num must be greater than 0.");
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

void InferenceRuntimeOptions::UseOpenVINOBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kOpenVINO),
           "OpenVINO backend is not available.");
  backend = InferenceBackendType::kOpenVINO;
}

void InferenceRuntimeOptions::SetTensorRTInputShape(
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

void InferenceRuntimeOptions::EnableTensorRTFP16() { trt_enable_fp16 = true; }

void InferenceRuntimeOptions::DisableTensorRTFP16() { trt_enable_fp16 = false; }

void InferenceRuntimeOptions::SetTensorRTCacheFile(
    const std::string& cache_file_path) {
  trt_serialize_file = cache_file_path;
}

bool InferenceRuntime::Init(const InferenceRuntimeOptions& input_options) {
  options = input_options;
  if (options.model_format == InferenceFrontendType::kAuto) {
    options.model_format = GuessModelFormat(options.model_file);
  }
  if (options.backend == InferenceBackendType::kUnknown) {
    if (IsBackendAvailable(InferenceBackendType::kONNXRuntime)) {
      options.backend = InferenceBackendType::kONNXRuntime;
    } else if (IsBackendAvailable(InferenceBackendType::kOpenVINO)) {
      options.backend = InferenceBackendType::kOpenVINO;
    } else {
      AINFO_F("Please define backend in RuntimeOption, current it's {}.",
              ToString(options.backend));
      return false;
    }
  }

  if (options.backend == InferenceBackendType::kONNXRuntime) {
    ACHECK_F(options.device == InferenceDeviceType::kCPU ||
                 options.device == InferenceDeviceType::kGPU,
             "{} only supports {} and {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU),
             ToString(InferenceDeviceType::kGPU));
    CreateONNXRuntimeBackend();
  } else if (options.backend == InferenceBackendType::kTensorRT) {
    ACHECK_F(options.device == InferenceDeviceType::kGPU,
             "{} only supports {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kGPU));
    CreateTensorRTBackend();
  } else if (options.backend == InferenceBackendType::kOpenVINO) {
    ACHECK_F(options.device == InferenceDeviceType::kCPU,
             "{} only supports {}.", ToString(options.backend),
             ToString(InferenceDeviceType::kCPU));
    CreateOpenVINOBackend();
  } else {
    AINFO_F("Unknown backend: {}.", ToString(options.backend));
    return false;
  }

  AINFO_F("Runtime initialized with {} in {}.", ToString(options.backend),
          ToString(options.device));

  return true;
}

void InferenceRuntime::CreateOpenVINOBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kOpenVINO),
           "OpenVINO backend is not available.");
  AFATAL_F("OpenVINO backend is not supported yet.");
}

void InferenceRuntime::CreateONNXRuntimeBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kONNXRuntime),
           "ONNXRuntime backend is not available.");
  AFATAL_F("OpenVINO backend is not supported yet.");
}

void InferenceRuntime::CreateTensorRTBackend() {
  ACHECK_F(IsBackendAvailable(InferenceBackendType::kTensorRT),
           "TensorRT backend is not available.");
  AFATAL_F("OpenVINO backend is not supported yet.");
}

}  // namespace inference
}  // namespace zetton
