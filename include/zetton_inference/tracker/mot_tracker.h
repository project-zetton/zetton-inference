#include "zetton_inference/interface/base_object_tracker.h"

namespace zetton {
namespace inference {
class MotTracker : public BaseObjectTracker {
  MotTracker() = default;
  ~MotTracker() override = default;

  void Infer() override{};
  bool Track() override { return true; };

};

}  // namespace inference
}  // namespace zetton