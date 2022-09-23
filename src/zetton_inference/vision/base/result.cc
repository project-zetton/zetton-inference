#include "zetton_inference/vision/base/result.h"

namespace zetton {
namespace inference {
namespace vision {

void ClassificationResult::Clear() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
}

std::string ClassificationResult::ToString() {
  std::string out;
  out = "ClassificationResult(\nlabel_ids: ";
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

DetectionResult::DetectionResult(const DetectionResult& res) : BaseResult(res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
}

void DetectionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
}

void DetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  label_ids.reserve(size);
}

void DetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  label_ids.resize(size);
}

std::string DetectionResult::ToString() {
  std::string out;
  out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out += std::to_string(boxes[i][0]) + "," + std::to_string(boxes[i][1]) +
           ", " + std::to_string(boxes[i][2]) + ", " +
           std::to_string(boxes[i][3]) + ", " + std::to_string(scores[i]) +
           ", " + std::to_string(label_ids[i]);
    out += "\n";
  }
  return out;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
