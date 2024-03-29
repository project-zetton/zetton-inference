#include "zetton_inference/vision/base/result.h"

#include <absl/strings/str_join.h>
#include <fmt/format.h>

namespace zetton {
namespace inference {
namespace vision {

void ClassificationResult::Clear() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
}

std::string ClassificationResult::ToString() {
  std::string out = "ClassificationResult: [label_id, score]\n";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out += fmt::format("-> [{}, {}]\n", label_ids[i], scores[i]);
  }
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
  std::string out =
      "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out +=
        fmt::format("-> [{}, {}, {}, {}, {}, {}]\n", boxes[i][0], boxes[i][1],
                    boxes[i][2], boxes[i][3], scores[i], label_ids[i]);
  }
  return out;
}

ReIDResult::ReIDResult(const ReIDResult& res) : BaseResult(res) {
  features.assign(res.features.begin(), res.features.end());
}

void ReIDResult::Clear() { std::vector<cv::Mat>().swap(features); }

void ReIDResult::Reserve(int size) { features.reserve(size); }

void ReIDResult::Resize(int size) { features.resize(size); }

TrackingResult::TrackingResult() : DetectionResult() {
  type = ResultType::kTracking;
}

TrackingResult::TrackingResult(const TrackingResult& res)
    : DetectionResult(res) {
  features.assign(res.features.begin(), res.features.end());
  tracking_ids.assign(res.tracking_ids.begin(), res.tracking_ids.end());
}

void TrackingResult::Clear() {
  DetectionResult::Clear();
  std::vector<cv::Mat>().swap(features);
  std::vector<int32_t>().swap(tracking_ids);
}

void TrackingResult::Reserve(int size) {
  DetectionResult::Reserve(size);
  features.reserve(size);
  tracking_ids.reserve(size);
}

void TrackingResult::Resize(int size) {
  DetectionResult::Resize(size);
  features.resize(size);
  tracking_ids.resize(size);
}

std::string TrackingResult::ToString() {
  std::string out =
      "Tracking: [tracking_id, xmin, ymin, xmax, ymax, score, label_id]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out += fmt::format("-> [{}, {}, {}, {}, {}, {}, {}]\n", tracking_ids[i],
                       boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                       scores[i], label_ids[i]);
  }
  return out;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
