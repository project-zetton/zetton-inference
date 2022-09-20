#include "zetton_inference/base/frame/syncedmem.h"

#include <zetton_common/log/log.h>

namespace zetton {
namespace inference {

SyncedMemory::SyncedMemory(bool use_cuda)
    : cpu_ptr_(nullptr),
      gpu_ptr_(nullptr),
      size_(0),
      head_(UNINITIALIZED),
      own_cpu_data_(false),
      cpu_malloc_use_cuda_(use_cuda),
      own_gpu_data_(false),
      device_(-1) {
#if USE_GPU == 1
#ifdef PERCEPTION_DEBUG
  BASE_CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size, bool use_cuda)
    : cpu_ptr_(nullptr),
      gpu_ptr_(nullptr),
      size_(size),
      head_(UNINITIALIZED),
      own_cpu_data_(false),
      cpu_malloc_use_cuda_(use_cuda),
      own_gpu_data_(false),
      device_(-1) {
#if USE_GPU == 1
#ifdef PERCEPTION_DEBUG
  BASE_CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    ZettonFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#if USE_GPU == 1
  if (gpu_ptr_ && own_gpu_data_) {
    BASE_CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // USE_GPU
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
    case UNINITIALIZED:
      ZettonMallocHost(&cpu_ptr_, size_, cpu_malloc_use_cuda_);
      if (cpu_ptr_ == nullptr) {
        AERROR << "cpu_ptr_ is null";
        return;
      }
      memset(cpu_ptr_, 0, size_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case HEAD_AT_GPU:
#if USE_GPU == 1
      if (cpu_ptr_ == nullptr) {
        ZettonMallocHost(&cpu_ptr_, size_, cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
      }
      BASE_CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault));
      head_ = SYNCED;
#else
      NOT_IMPLEMENTED;
#endif
      break;
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#if USE_GPU == 1
  switch (head_) {
    case UNINITIALIZED:
      BASE_CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      BASE_CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
      head_ = HEAD_AT_GPU;
      own_gpu_data_ = true;
      break;
    case HEAD_AT_CPU:
      if (gpu_ptr_ == nullptr) {
        BASE_CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        own_gpu_data_ = true;
      }
      BASE_CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault));
      head_ = SYNCED;
      break;
    case HEAD_AT_GPU:
    case SYNCED:
      break;
  }
#else
  NOT_IMPLEMENTED;
#endif
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  ACHECK_NOTNULL(data);
  if (own_cpu_data_) {
    ZettonFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#if USE_GPU == 1
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NOT_IMPLEMENTED;
  return nullptr;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#if USE_GPU == 1
  ACHECK_NOTNULL(data);
  if (own_gpu_data_) {
    BASE_CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NOT_IMPLEMENTED;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#if USE_GPU == 1
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NOT_IMPLEMENTED;
  return nullptr;
#endif
}

#if USE_GPU == 1
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  ACHECK_EQ(head_, HEAD_AT_CPU);
  if (gpu_ptr_ == nullptr) {
    BASE_CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  BASE_CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#if USE_GPU == 1
#ifdef PERCEPTION_DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK_EQ(device, device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    BASE_CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK_EQ(attributes.device, device_);
  }
#endif
#endif
}

}  // namespace inference
}  // namespace zetton
