#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zetton {
namespace inference {

enum class VisionResultType {
  kUnknown,
  kClassification,
  kDetection,
  kTracking,
};

struct BaseVisionResult {
  VisionResultType type = VisionResultType::kUnknown;
};

struct ClassificationVisionResult : public BaseVisionResult {
 public:
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  VisionResultType type = VisionResultType::kClassification;

 public:
  void Clear();
  std::string ToString();
};

struct DetectionVisionResult : public BaseVisionResult {
 public:
  /// \brief bounding boxes (xmin, ymin, xmax, ymax)
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  VisionResultType type = VisionResultType::kDetection;

 public:
  DetectionVisionResult() {}
  DetectionVisionResult(const DetectionVisionResult& res);

 public:
  void Clear();
  void Reserve(int size);
  void Resize(int size);
  std::string ToString();
};

}  // namespace inference
}  // namespace zetton
