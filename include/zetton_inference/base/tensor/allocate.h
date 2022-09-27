#pragma once

#include "zetton_inference/base/common.h"

namespace zetton {
namespace inference {

/// \brief allocate host memory
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

/// \brief free host memory
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
/// \brief allocate device memory
inline void ZettonMallocDevice(void** ptr, size_t size) {
  BASE_CUDA_CHECK(cudaMalloc(ptr, size));
}

/// \brief free device memory
inline void ZettonFreeDevice(void* ptr) { BASE_CUDA_CHECK(cudaFree(ptr)); }
#endif

}  // namespace inference
}  // namespace zetton
