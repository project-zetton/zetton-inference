#pragma once

#include <vector>

#include "zetton_inference/base/frame/data_provider.h"
#include "zetton_inference/base/object/object.h"

namespace zetton {
namespace inference {

struct CameraFrame {
  // timestamp
  double timestamp = 0.0;
  // frame sequence id
  int frame_id = 0;
  // data provider
  DataProvider *data_provider = nullptr;
  // tracker proposed objects
  std::vector<ObjectPtr> proposed_objects;
  // segmented objects
  std::vector<ObjectPtr> detected_objects;
  // tracked objects
  std::vector<ObjectPtr> tracked_objects;
};

}  // namespace inference
}  // namespace zetton
