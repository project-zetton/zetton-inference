#pragma once

#if USE_GPU == 1
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "zetton_common/log/log.h"

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED AFATAL_F("Not implemented yet.")
#endif

namespace zetton {
namespace inference {

#if USE_GPU == 1
inline void GPUAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    AERROR_F("GPUassert: {} {} {}", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}
#define BASE_CUDA_CHECK(condition) \
  { zetton::inference::GPUAssert((condition), __FILE__, __LINE__); }
#endif

}  // namespace inference
}  // namespace zetton
