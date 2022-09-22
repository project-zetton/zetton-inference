#pragma once

#include <fmt/format.h>

#include <string>
#include <vector>

#include "zetton_inference/base/type.h"

namespace zetton {
namespace inference {

struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  InferenceDataType dtype;
};

std::string ToString(const TensorInfo& info);

}  // namespace inference
}  // namespace zetton
