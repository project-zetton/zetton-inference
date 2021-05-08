#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "zetton_common/util/registerer.h"
#include "zetton_inference/interface/base_inference.h"

namespace zetton {
namespace inference {

class BaseObjectTracker : public BaseInference {
 public:
  void Infer() override = 0;
  virtual bool Track() = 0;

  ZETTON_REGISTER_REGISTERER(BaseObjectTracker);
#define ZETTON_REGISTER_OBJECT_TRACKER(name) \
  ZETTON_REGISTER_CLASS(BaseObjectTracker, name)
};

}  // namespace inference
}  // namespace zetton