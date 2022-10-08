#include "zetton_inference/base/tensor/tensor_info.h"

namespace zetton {
namespace inference {

std::string ToString(const TensorInfo& info) {
  return fmt::format("TensorInfo(name: {}, shape: [{}], dtype: {})", info.name,
                     fmt::join(info.shape, ", "), ToString(info.dtype));
}

}  // namespace inference
}  // namespace zetton
