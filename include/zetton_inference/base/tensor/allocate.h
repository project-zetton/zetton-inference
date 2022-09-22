#pragma once

#include "zetton_inference/base/common.h"

namespace zetton {
namespace inference {

inline void ZettonMallocHost(void** ptr, size_t size, bool use_cuda) {
#if USE_GPU == 1
  if (use_cuda) {
    BASE_CUDA_CHECK(cudaMallocHost(ptr, size));
    return;
  }
#endif
  *ptr = malloc(size);
  ACHECK_NOTNULL_F(*ptr, "host allocation of size {} failed", size);
}

inline void ZettonFreeHost(void* ptr, bool use_cuda) {
#if USE_GPU == 1
  if (use_cuda) {
    BASE_CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}

#if USE_GPU == 1
inline void ZettonMallocDevice(void** ptr, size_t size) {
  BASE_CUDA_CHECK(cudaMalloc(ptr, size));
}

inline void ZettonFreeDevice(void* ptr) { BASE_CUDA_CHECK(cudaFree(ptr)); }
#endif

}  // namespace inference
}  // namespace zetton
