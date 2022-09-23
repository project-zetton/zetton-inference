#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zetton {
namespace inference {

namespace vision {

enum class ResultType {
  kUnknown,
  kClassification,
  kDetection,
  kTracking,
};

struct BaseResult {
  ResultType type = ResultType::kUnknown;
};

struct ClassificationResult : public BaseResult {
 public:
  std::vector<int32_t> label_ids;
  std::vector<float> scores;
  ResultType type = ResultType::kClassification;

 public:
  void Clear();
  std::string ToString();
};

struct DetectionResult : public BaseResult {
 public:
  /// \brief bounding boxes (xmin, ymin, xmax, ymax)
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  ResultType type = ResultType::kDetection;

 public:
  DetectionResult() {}
  DetectionResult(const DetectionResult& res);

 public:
  void Clear();
  void Reserve(int size);
  void Resize(int size);
  std::string ToString();
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
