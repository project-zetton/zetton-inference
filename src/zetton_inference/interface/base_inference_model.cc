#include "zetton_inference/interface/base_inference_model.h"

#include <memory>

#include "zetton_common/util/perf.h"
#include "zetton_inference/base/runtime.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

bool BaseInferenceModel::InitRuntime() {
  // check inference frontend
  ACHECK_F(CheckModelFormat(runtime_options.model_file,
                            runtime_options.model_format),
           "Failed to check model format.");

  // check whether the model is initialized
  if (runtime_initialized_) {
    AERROR_F("Runtime has been initialized already.");
    return false;
  }

  // check inference backend
  if (runtime_options.backend != InferenceBackendType::kUnknown) {
    if (!IsBackendAvailable(runtime_options.backend)) {
      AERROR_F("Backend {} is not available.",
               ToString(runtime_options.backend));
      return false;
    }

    bool use_gpu = (runtime_options.device == InferenceDeviceType::kGPU);
#ifndef USE_GPU
    use_gpu = false;
#endif
    bool use_npu = (runtime_options.device == InferenceDeviceType::kNPU);

    // check whether the model is supported by the set backend
    bool is_supported = false;
    if (use_gpu) {
      for (auto& item : valid_gpu_backends) {
        if (item == runtime_options.backend) {
          is_supported = true;
          break;
        }
      }
    } else if (use_npu) {
      for (auto& item : valid_npu_backends) {
        if (item == runtime_options.backend) {
          is_supported = true;
          break;
        }
      }
    } else {
      for (auto& item : valid_cpu_backends) {
        if (item == runtime_options.backend) {
          is_supported = true;
          break;
        }
      }
    }

    if (is_supported) {
      // just use the set backend
      AINFO_F("Use {} backend.", ToString(runtime_options.backend));
      runtime_ = std::make_unique<InferenceRuntime>();
      if (!runtime_->Init(runtime_options)) {
        return false;
      }
      runtime_initialized_ = true;
      return true;
    } else {
      // try to find a supported backend
      AWARN_F("Invalid backend {} for model {}.",
              ToString(runtime_options.backend), Name());
      if (use_gpu) {
        ACHECK_F(valid_gpu_backends.size() > 0,
                 "No available GPU backend found.");
        AWARN_F("Use {} backend by default.", ToString(valid_gpu_backends[0]));
      } else if (use_npu) {
        ACHECK_F(valid_npu_backends.size() > 0,
                 "No available NPU backend found.");
        AWARN_F("Use {} backend by default.", ToString(valid_npu_backends[0]));
      } else {
        ACHECK_F(valid_cpu_backends.size() > 0,
                 "No available CPU backend found.");
        AWARN_F("Use {} backend by default.", ToString(valid_cpu_backends[0]));
      }
    }
  }

  // check device type and create backend
  if (runtime_options.device == InferenceDeviceType::kCPU) {
    return CreateCpuBackend();
  } else if (runtime_options.device == InferenceDeviceType::kGPU) {
#ifdef USE_GPU
    return CreateGpuBackend();
#else
    AERROR_F("The compiled library doesn't support GPU now.");
    return false;
#endif
  } else if (runtime_options.device == InferenceDeviceType::kNPU) {
    AERROR_F("NPU is not supported now.");
    return false;
  }
  AERROR_F("Invalid device type {}.", ToString(runtime_options.device));
  return false;
}

bool BaseInferenceModel::CreateCpuBackend() {
  if (valid_cpu_backends.size() == 0) {
    AERROR_F("No available CPU backend found for model {}.", Name());
    return false;
  }

  for (auto& valid_cpu_backend : valid_cpu_backends) {
    if (!IsBackendAvailable(valid_cpu_backend)) {
      continue;
    }
    runtime_options.backend = valid_cpu_backend;
    runtime_ = std::make_unique<InferenceRuntime>();
    if (!runtime_->Init(runtime_options)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  AERROR_F("No available CPU backend found for model {}.", Name());
  return false;
}

bool BaseInferenceModel::CreateGpuBackend() {
  if (valid_gpu_backends.size() == 0) {
    AERROR_F("No available GPU backend found for model {}.", Name());
    return false;
  }

  for (auto& valid_gpu_backend : valid_gpu_backends) {
    if (!IsBackendAvailable(valid_gpu_backend)) {
      continue;
    }
    runtime_options.backend = valid_gpu_backend;
    runtime_ = std::make_unique<InferenceRuntime>();
    if (!runtime_->Init(runtime_options)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  AERROR_F("No available GPU backend found for model {}.", Name());
  return false;
}

bool BaseInferenceModel::CreateNpuBackend() {
  if (valid_npu_backends.size() == 0) {
    AERROR_F("No available NPU backend found for model {}.", Name());
    return false;
  }

  for (auto& valid_npu_backend : valid_npu_backends) {
    if (!IsBackendAvailable(valid_npu_backend)) {
      continue;
    }
    runtime_options.backend = valid_npu_backend;
    runtime_ = std::make_unique<InferenceRuntime>();
    if (!runtime_->Init(runtime_options)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  AERROR_F("No available NPU backend found for model {}.", Name());
  return false;
}

bool BaseInferenceModel::Infer(std::vector<Tensor>& input_tensors,
                               std::vector<Tensor>* output_tensors) {
  // start recording inference time
  zetton::common::TimeCounter tc;
  if (enable_record_time_of_runtime_) {
    tc.Start();
  }

  // do inference
  auto ret = runtime_->Infer(input_tensors, output_tensors);

  // end recording infenrence time
  if (enable_record_time_of_runtime_) {
    tc.End();
    if (time_of_runtime_.size() > 50000) {
      AWARN_F("Already record 50000 inference times, stop recording.");
      enable_record_time_of_runtime_ = false;
    }
    time_of_runtime_.push_back(tc.Duration());
  }

  return ret;
}

std::map<std::string, double> BaseInferenceModel::PrintStatsInfoOfRuntime() {
  std::map<std::string, double> stats_info_of_runtime_dict;

  if (time_of_runtime_.size() < 10) {
    AWARN_F("Too few inference times to print stats info.");
  }
  double warmup_time = 0.0;
  double remain_time = 0.0;
  std::size_t warmup_iter = time_of_runtime_.size() / 5;
  for (std::size_t i = 0; i < time_of_runtime_.size(); ++i) {
    if (i < warmup_iter) {
      warmup_time += time_of_runtime_[i];
    } else {
      remain_time += time_of_runtime_[i];
    }
  }
  double avg_time =
      remain_time / static_cast<double>(time_of_runtime_.size() - warmup_iter);

  AINFO_F("============= Runtime Stats Info({}) =============", Name());
  AINFO_F("Total iterations: {}", time_of_runtime_.size());
  AINFO_F("Total time of runtime: {}s.", warmup_time + remain_time);
  AINFO_F("Warmup iterations: {}.", warmup_iter);
  AINFO_F("Total time of runtime in warmup step: {}s.", warmup_time);
  AINFO_F("Average time of runtime exclude warmup step: {}ms.",
          avg_time * 1000);
  AINFO_F("Average FPS exclude warmup step: {}.", 1.0 / avg_time);
  AINFO_F("============= Runtime Stats Info({}) =============", Name());

  stats_info_of_runtime_dict["total_time"] = warmup_time + remain_time;
  stats_info_of_runtime_dict["warmup_time"] = warmup_time;
  stats_info_of_runtime_dict["remain_time"] = remain_time;
  stats_info_of_runtime_dict["warmup_iter"] = static_cast<double>(warmup_iter);
  stats_info_of_runtime_dict["avg_time"] = avg_time;
  stats_info_of_runtime_dict["iterations"] =
      static_cast<double>(time_of_runtime_.size());

  return stats_info_of_runtime_dict;
}

}  // namespace inference
}  // namespace zetton
