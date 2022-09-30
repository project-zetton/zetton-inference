#pragma once

#include "zetton_inference/vision/base/result.h"

namespace zetton {
namespace inference {
namespace vision {

struct BaseVisionTrackerParams {};

class BaseVisionTracker {
 public:
  BaseVisionTracker() = default;
  ~BaseVisionTracker() = default;

 public:
  /// \brief update tracks wiht the given detection results
  /// \param detections detection results
  /// \param tracks output tracking results
  virtual bool Update(const DetectionResult &detections,
                      TrackingResult &tracks) = 0;

  /// \brief update tracks wiht the given detection and ReID results
  /// \param detections detection results
  /// \param features ReID results
  /// \param tracks output tracking results
  virtual bool Update(const DetectionResult &detections,
                      const ReIDResult &features, TrackingResult &tracks) = 0;

  /// \brief get model name
  virtual std::string Name() = 0;

 public:
  /// \brief get params
  virtual BaseVisionTrackerParams *GetParams() = 0;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
