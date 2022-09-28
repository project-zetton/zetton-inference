#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zetton {
namespace inference {
namespace vision {

/// \brief Types of all vision inference results
enum class ResultType {
  kUnknown,
  kClassification,
  kDetection,
  kTracking,
  kReID,
};

/// \brief Base class for all vision inference results
struct BaseResult {
  ResultType type = ResultType::kUnknown;
};

/// \brief Classification result from vision model
struct ClassificationResult : public BaseResult {
 public:
  /// \brief labels of classification result
  std::vector<int32_t> label_ids;
  /// \brief scores of classification result
  std::vector<float> scores;
  /// \brief result type
  ResultType type = ResultType::kClassification;

 public:
  /// \brief clear the result
  void Clear();
  /// \brief convert the result to string
  std::string ToString();
};

/// \brief Detection result from vision model
struct DetectionResult : public BaseResult {
 public:
  /// \brief bounding boxes (xmin, ymin, xmax, ymax)
  std::vector<std::array<float, 4>> boxes;
  /// \brief labels of detection result
  std::vector<int32_t> label_ids;
  /// \brief scores of detection result
  std::vector<float> scores;
  /// \brief result type
  ResultType type = ResultType::kDetection;

 public:
  /// \brief default constructor
  DetectionResult() = default;
  /// \brief copy the result from another detection result
  DetectionResult(const DetectionResult& res);

 public:
  /// \brief clear the result
  void Clear();
  /// \brief reserve memory for the result
  /// \param num number of detection results
  void Reserve(int size);
  /// \brief resize the result
  /// \param num number of detection results
  void Resize(int size);
  /// \brief convert the result to string
  std::string ToString();
};

/// \brief ReID result from vision model
struct ReIDResult : public BaseResult {
 public:
  /// \brief extracted features from ReID model
  std::vector<std::vector<float>> features;
  /// \brief result type
  ResultType type = ResultType::kReID;

 public:
  /// \brief default constructor
  ReIDResult() = default;
  /// \brief copy the result from another ReID result
  ReIDResult(const ReIDResult& res);

 public:
  /// \brief clear the result
  void Clear();
  /// \brief reserve memory for the result
  /// \param num number of ReID results
  void Reserve(int size);
  /// \brief resize the result
  /// \param num number of ReID results
  void Resize(int size);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
