#pragma once

#include <string>

namespace zetton {
namespace inference {

class BaseInference {
 public:
  BaseInference() = default;
  virtual ~BaseInference() = default;

 public:
  virtual std::string Name() const = 0;

 public:
  BaseInference(const BaseInference&) = delete;
  BaseInference& operator=(const BaseInference&) = delete;
};

}  // namespace inference
}  // namespace zetton
