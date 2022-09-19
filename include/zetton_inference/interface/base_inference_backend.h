#pragma once

#include <map>
#include <memory>
#include <vector>

#include "zetton_common/util/registerer.h"

namespace zetton {
namespace inference {

class BaseInferenceBackend {
 public:
  BaseInferenceBackend() = default;
  virtual ~BaseInferenceBackend() = default;

 public:
  virtual bool Init(const std::map<std::string, std::vector<int>> &shapes) = 0;
  virtual void Infer() = 0;

 public:
  void SetMaxBatchSize(const int &batch_size) { max_batch_size_ = batch_size; }
  void SetGpuId(const int &gpu_id) { gpu_id_ = gpu_id; }

 protected:
  int max_batch_size_ = 1;
  int gpu_id_ = -1;
};

ZETTON_REGISTER_REGISTERER(BaseInferenceBackend)
#define ZETTON_REGISTER_INFERENCE_BACKEND(name) \
  ZETTON_REGISTER_CLASS(BaseInferenceBackend, name)

}  // namespace inference
}  // namespace zetton
