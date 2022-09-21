#include "zetton_inference/vision/common/result.h"

namespace zetton {
namespace inference {

void ClassificationVisionResult::Clear() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
}

std::string ClassificationVisionResult::ToString() {
  std::string out;
  out = "ClassificationVisionResult(\nlabel_ids: ";
  for (int label_id : label_ids) {
    out += std::to_string(label_id) + ", ";
  }
  out += "\nscores: ";
  for (float score : scores) {
    out += std::to_string(score) + ", ";
  }
  out += "\n)";
  return out;
}

DetectionVisionResult::DetectionVisionResult(const DetectionVisionResult& res)
    : BaseVisionResult(res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
}

void DetectionVisionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
}

void DetectionVisionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  label_ids.reserve(size);
}

void DetectionVisionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  label_ids.resize(size);
}

std::string DetectionVisionResult::ToString() {
  std::string out;
  out = "DetectionVisionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out += std::to_string(boxes[i][0]) + "," + std::to_string(boxes[i][1]) +
           ", " + std::to_string(boxes[i][2]) + ", " +
           std::to_string(boxes[i][3]) + ", " + std::to_string(scores[i]) +
           ", " + std::to_string(label_ids[i]);
    out += "\n";
  }
  return out;
}

}  // namespace inference
}  // namespace zetton
