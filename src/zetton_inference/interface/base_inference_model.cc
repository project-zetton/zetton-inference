#include "zetton_inference/interface/base_inference_model.h"

#include <memory>

#include "zetton_common/util/perf.h"
#include "zetton_inference/base/runtime.h"
#include "zetton_inference/base/type.h"
#include "zetton_inference/util/runtime_util.h"

namespace zetton {
namespace inference {

bool BaseInferenceModel::InitRuntime() {
  ACHECK_F(CheckModelFormat(runtime_options.model_file,
                            runtime_options.model_format),
           "ModelFormatCheck Failed.");
  if (runtime_initialized_) {
    AERROR_F("The model is already initialized, cannot be initliazed again.");
    return false;
  }
  if (runtime_options.backend != InferenceBackendType::kUnknown) {
    if (!IsBackendAvailable(runtime_options.backend)) {
      AERROR_F("{} is not compiled with current library.",
               ToString(runtime_options.backend));
      return false;
    }

    bool use_gpu = (runtime_options.device == InferenceDeviceType::kGPU);
#ifndef USE_GPU
    use_gpu = false;
#endif

    // whether the model is supported by the setted backend
    bool is_supported = false;
    if (use_gpu) {
      for (auto& item : valid_gpu_backends) {
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
      runtime_ = std::make_unique<InferenceRuntime>();
      if (!runtime_->Init(runtime_options)) {
        return false;
      }
      runtime_initialized_ = true;
      return true;
    } else {
      AWARN_F("{} is not supported with backend {}.", Name(),
              ToString(runtime_options.backend));
      if (use_gpu) {
        ACHECK_F(valid_gpu_backends.size() > 0,
                 "There's no valid gpu backend for {}.", Name());
        AWARN_F("Choose {} for model inference.",
                ToString(valid_gpu_backends[0]));
      } else {
        ACHECK_F(valid_cpu_backends.size() > 0,
                 "There's no valid cpu backend for {}.", Name());
        AWARN_F("Choose {} for model inference.",
                ToString(valid_cpu_backends[0]));
      }
    }
  }

  if (runtime_options.device == InferenceDeviceType::kCPU) {
    return CreateCpuBackend();
  } else if (runtime_options.device == InferenceDeviceType::kGPU) {
#ifdef USE_GPU
    return CreateGpuBackend();
#else
    AERROR_F("The compiled library doesn't support GPU now.");
    return false;
#endif
  }
  AERROR_F("Only support CPU/GPU now.");
  return false;
}

bool BaseInferenceModel::CreateCpuBackend() {
  if (valid_cpu_backends.size() == 0) {
    AERROR_F("There's no valid cpu backends for model: {}.", Name());
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
  AERROR_F("Found no valid backend for model: {}.", Name());
  return false;
}

bool BaseInferenceModel::CreateGpuBackend() {
  if (valid_gpu_backends.size() == 0) {
    AERROR_F("There's no valid gpu backends for model: {}.", Name());
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
  AERROR_F("Cannot find an available gpu backend to load this model.");
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
      AWARN_F(
          "There are already 50000 records of runtime, will force to disable "
          "record time of runtime now.");
      enable_record_time_of_runtime_ = false;
    }
    time_of_runtime_.push_back(tc.Duration());
  }

  return ret;
}

std::map<std::string, float> BaseInferenceModel::PrintStatisInfoOfRuntime() {
  std::map<std::string, float> statis_info_of_runtime_dict;

  if (time_of_runtime_.size() < 10) {
    AWARN_F(
        "PrintStatisInfoOfRuntime require the runtime ran 10 times at "
        "least, but now you only ran {} times.",
        time_of_runtime_.size());
  }
  double warmup_time = 0.0;
  double remain_time = 0.0;
  int warmup_iter = time_of_runtime_.size() / 5;
  for (size_t i = 0; i < time_of_runtime_.size(); ++i) {
    if (i < warmup_iter) {
      warmup_time += time_of_runtime_[i];
    } else {
      remain_time += time_of_runtime_[i];
    }
  }
  double avg_time = remain_time / (time_of_runtime_.size() - warmup_iter);

  AINFO_F("============= Runtime Statis Info({}) =============", Name());
  AINFO_F("Total iterations: {}", time_of_runtime_.size());
  AINFO_F("Total time of runtime: {}s.", warmup_time + remain_time);
  AINFO_F("Warmup iterations: {}.", warmup_iter);
  AINFO_F("Total time of runtime in warmup step: {}s.", warmup_time);
  AINFO_F("Average time of runtime exclude warmup step: {}ms.",
          avg_time * 1000);

  statis_info_of_runtime_dict["total_time"] = warmup_time + remain_time;
  statis_info_of_runtime_dict["warmup_time"] = warmup_time;
  statis_info_of_runtime_dict["remain_time"] = remain_time;
  statis_info_of_runtime_dict["warmup_iter"] = warmup_iter;
  statis_info_of_runtime_dict["avg_time"] = avg_time;
  statis_info_of_runtime_dict["iterations"] = time_of_runtime_.size();

  return statis_info_of_runtime_dict;
}

}  // namespace inference
}  // namespace zetton
