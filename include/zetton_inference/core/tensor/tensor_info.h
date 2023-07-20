#pragma once

#include <fmt/format.h>

#include <string>
#include <vector>

#include "zetton_inference/core/type.h"

namespace zetton {
namespace inference {

struct TensorInfo {
  /// \brief name of tensor
  std::string name;
  /// \brief shape of tensor
  std::vector<int> shape;
  /// \brief data type of tensor
  InferenceDataType dtype;
};

/// \brief convert tensor info to string
std::string ToString(const TensorInfo& info);

}  // namespace inference
}  // namespace zetton
