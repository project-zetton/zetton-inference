#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "zetton_common/util/registerer.h"
#include "zetton_inference/interface/base_inference.h"

namespace zetton {
namespace inference {

struct ObjectDetectionResult {
  ObjectDetectionResult(cv::Rect bbox, int type, float prob)
      : bbox(std::move(bbox)), type(type), prob(prob) {}

  /**
   * @brief Draw boudning boxes on given image
   *
   * @param frame Input image
   */
  inline void Draw(cv::Mat& frame) {
    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << type << "@" << prob;
    cv::putText(frame, stream.str(), cv::Point(bbox.x, bbox.y - 5), 0, 0.5,
                cv::Scalar(0, 0, 255), 2);
  }

  inline friend std::ostream& operator<<(std::ostream& os,
                                         const ObjectDetectionResult& result) {
    os << result.type << "@" << result.prob << ":" << result.bbox;
    return os;
  }

  cv::Rect bbox;
  int type = -1;
  float prob = 0.0;
};

typedef std::vector<ObjectDetectionResult> ObjectDetectionResults;
typedef std::vector<ObjectDetectionResults> BatchObjectDetectionResults;

class BaseObjectDetector : public BaseInference {
 public:
  void Infer() override = 0;
  /**
   * @brief Detect objects in given image
   *
   * @param frame Input image
   * @param result Output object detection results
   * @return `true`  if detection is successful
   */
  virtual bool Detect(const cv::Mat& frame, ObjectDetectionResults& result) = 0;

  ZETTON_REGISTER_REGISTERER(BaseObjectDetector)
#define ZETTON_REGISTER_OBJECT_DETECTOR(name) \
  ZETTON_REGISTER_CLASS(BaseObjectDetector, name)
};

}  // namespace inference
}  // namespace zetton
