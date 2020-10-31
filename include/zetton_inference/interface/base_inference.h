#pragma once

namespace zetton {
namespace inference {

class BaseInference {
 public:
  BaseInference() = default;
  virtual ~BaseInference() = default;

  virtual bool Init() = 0;
  virtual void Infer() = 0;
};

}  // namespace inference
}  // namespace zetton
