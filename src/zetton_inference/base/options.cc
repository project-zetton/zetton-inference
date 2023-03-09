#include "zetton_inference/base/options.h"

#include "zetton_common/log/log.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

void InferenceRuntimeOptions::SetModelPath(
    const std::string& model_path, const std::string& params_path,
    const InferenceFrontendType& input_model_format) {
  model_file = model_path;
  params_file = params_path;
  model_format = input_model_format;
  model_from_memory = false;
}

void InferenceRuntimeOptions::SetModelBuffer(
    const std::string& model_buffer, const std::string& params_buffer,
    const InferenceFrontendType& input_model_format) {
  model_file = model_buffer;
  params_file = params_buffer;
  model_format = input_model_format;
  model_from_memory = true;
}

void InferenceRuntimeOptions::SetEncryptionKey(
    const std::string& input_encryption_key) {
#ifdef ENABLE_ENCRYPTION
  encryption_key = input_encryption_key;
#else
  AERROR_F("Please compile with ENABLE_ENCRYPTION=ON.");
#endif
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

}  // namespace inference
}  // namespace zetton
