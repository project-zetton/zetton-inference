#pragma once

#include <cstddef>

namespace zetton {
namespace inference {
namespace vision {
namespace bytetrack {

int lapjv_internal(const std::size_t n, double *cost[], int *x, int *y);

}
}  // namespace vision
}  // namespace inference
}  // namespace zetton
