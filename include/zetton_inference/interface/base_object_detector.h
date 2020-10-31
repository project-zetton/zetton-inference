#pragma once

#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "zetton_common/util/registerer.h"
#include "zetton_inference/interface/base_inference.h"

namespace zetton {
namespace inference {

struct ObjectDetectionResult {
  cv::Rect bbox;
  int type = -1;
  float prob = 0.0;
  ObjectDetectionResult(cv::Rect bbox, int type, float prob)
      : bbox(std::move(bbox)), type(type), prob(prob) {}
};

typedef std::vector<ObjectDetectionResult> ObjectDetectionResults;
typedef std::vector<ObjectDetectionResults> BatchObjectDetectionResults;

class BaseObjectDetector : public BaseInference {
 public:
  void Infer() override = 0;
  virtual bool Detect(const cv::Mat& frame, ObjectDetectionResults& result) = 0;

  ZETTON_REGISTER_REGISTERER(BaseObjectDetector);
#define ZETTON_REGISTER_OBJECT_DETECTOR(name) \
  ZETTON_REGISTER_CLASS(BaseObjectDetector, name)
};

}  // namespace inference
}  // namespace zetton