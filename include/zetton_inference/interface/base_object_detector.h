#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

#include "zetton_common/util/registerer.h"
#include "zetton_inference/base/object/object.h"
#include "zetton_inference/interface/base_inference.h"

namespace zetton {
namespace inference {

struct ObjectDetectorInitOptions {};

struct ObjectDetectorOptions {};

class BaseObjectDetector : public BaseInference {
 public:
  BaseObjectDetector() = default;
  ~BaseObjectDetector() override = default;

 public:
  virtual bool Init(const ObjectDetectorInitOptions& options =
                        ObjectDetectorInitOptions()) = 0;

 public:
  /// \brief Detect objects in given image
  virtual bool Detect(const cv::Mat& frame,
                      std::vector<ObjectPtr>& results) = 0;
};

ZETTON_REGISTER_REGISTERER(BaseObjectDetector)
#define ZETTON_REGISTER_OBJECT_DETECTOR(name) \
  ZETTON_REGISTER_CLASS(BaseObjectDetector, name)

}  // namespace inference
}  // namespace zetton
