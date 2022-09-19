#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "zetton_common/util/registerer.h"
#include "zetton_inference/interface/base_inference.h"

namespace zetton {
namespace inference {

struct ObjectTrackerInitOptions {};

struct ObjectTrackerOptions {};

class BaseObjectTracker : public BaseInference {
 public:
  BaseObjectTracker() = default;
  ~BaseObjectTracker() override = default;

 public:
  virtual bool Init(
      const ObjectTrackerInitOptions& options = ObjectTrackerInitOptions()) = 0;

 public:
  virtual bool Track() = 0;
};

ZETTON_REGISTER_REGISTERER(BaseObjectTracker)
#define ZETTON_REGISTER_OBJECT_TRACKER(name) \
  ZETTON_REGISTER_CLASS(BaseObjectTracker, name)

}  // namespace inference
}  // namespace zetton
