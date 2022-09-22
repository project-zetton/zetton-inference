#include "zetton_inference/base/frame/data_provider.h"

#include <cwchar>

#include "zetton_inference/base/tensor/syncedmem.h"

#if USE_GPU == 1
#include <nppi.h>
#endif

namespace zetton {
namespace inference {

bool DataProvider::Init(const DataProviderInitOptions &options) {
  src_height_ = options.image_height;
  src_width_ = options.image_width;
  sensor_name_ = options.sensor_name;
  device_id_ = options.device_id;

#if USE_GPU == 1
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    AERROR << "Failed to set device to: " << device_id_;
    return false;
  }
#endif

  // Initialize uint8 blobs
  gray_.reset(new Image8U(src_height_, src_width_, ImageColorspace::GRAY));
  rgb_.reset(new Image8U(src_height_, src_width_, ImageColorspace::RGB));
  bgr_.reset(new Image8U(src_height_, src_width_, ImageColorspace::BGR));

  // Allocate CPU memory for uint8 blobs
  gray_->cpu_data();
  rgb_->cpu_data();
  bgr_->cpu_data();

  // Allocate GPU memory for uint8 blobs
  gray_->gpu_data();
  rgb_->gpu_data();
  bgr_->gpu_data();

  // Warm up nppi functions
  {
    bgr_ready_ = false;
    rgb_ready_ = true;
    gray_ready_ = false;
    to_bgr_image();
    bgr_ready_ = false;
    rgb_ready_ = false;
    gray_ready_ = true;
    to_bgr_image();
  }
  {
    bgr_ready_ = false;
    rgb_ready_ = false;
    gray_ready_ = true;
    to_rgb_image();
    bgr_ready_ = true;
    rgb_ready_ = false;
    gray_ready_ = false;
    to_rgb_image();
  }
  {
    bgr_ready_ = false;
    rgb_ready_ = true;
    gray_ready_ = false;
    to_gray_image();
    bgr_ready_ = true;
    rgb_ready_ = false;
    gray_ready_ = false;
    to_gray_image();
  }
  bgr_ready_ = false;
  rgb_ready_ = false;
  gray_ready_ = false;

  return true;
}

bool DataProvider::FillImageData(int rows, int cols, const uint8_t *data,
                                 const std::string &encoding) {
#if USE_GPU == 1
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    AERROR << "Failed to set device to: " << device_id_;
    return false;
  }
#endif

  gray_ready_ = false;
  rgb_ready_ = false;
  bgr_ready_ = false;

  bool success = false;

#if USE_GPU == 0  // copy to host memory
  AINFO << "Fill in CPU mode ...";
  if (encoding == "rgb8") {
    memcpy(rgb_->mutable_cpu_data(), data, rgb_->total() * sizeof(data[0]));
    rgb_ready_ = true;
    success = true;
  } else if (encoding == "bgr8") {
    memcpy(bgr_->mutable_cpu_data(), data, bgr_->total() * sizeof(data[0]));
    bgr_ready_ = true;
    success = true;
  } else if (encoding == "gray" || encoding == "y") {
    memcpy(gray_->mutable_cpu_data(), data, gray_->total() * sizeof(data[0]));
    gray_ready_ = true;
    success = true;
  } else {
    AERROR << "Unrecognized image encoding: " << encoding;
  }
#else  // copy to device memory directly
  AINFO << "Fill in GPU mode ...";
  if (encoding == "rgb8") {
    cudaMemcpy(rgb_->mutable_gpu_data(), data,
               rgb_->rows() * rgb_->width_step(), cudaMemcpyDefault);
    success = true;
    rgb_ready_ = true;
  } else if (encoding == "bgr8") {
    cudaMemcpy(bgr_->mutable_gpu_data(), data,
               bgr_->rows() * bgr_->width_step(), cudaMemcpyDefault);
    success = true;
    bgr_ready_ = true;
  } else if (encoding == "gray" || encoding == "y") {
    cudaMemcpy(gray_->mutable_gpu_data(), data,
               gray_->rows() * gray_->width_step(), cudaMemcpyDefault);
    success = true;
    gray_ready_ = true;
  } else {
    AERROR << "Unrecognized image encoding: " << encoding;
  }
#endif

  AINFO << "Done! (" << success << ")";
  return success;
}

bool DataProvider::GetImageBlob(const DataProviderImageOptions &options,
                                Blob<uint8_t> *blob) {
  Image8U image;
  if (!GetImage(options, &image)) {
    return false;
  }

#if USE_GPU == 1
  NppiSize roi;
  roi.height = image.rows();
  roi.width = image.cols();
  blob->Reshape({1, roi.height, roi.width, image.channels()});
  if (image.channels() == 1) {
    nppiCopy_8u_C1R(image.gpu_data(), image.width_step(),
                    blob->mutable_gpu_data(),
                    blob->count(2) * static_cast<int>(sizeof(uint8_t)), roi);
  } else {
    nppiCopy_8u_C3R(image.gpu_data(), image.width_step(),
                    blob->mutable_gpu_data(),
                    blob->count(2) * static_cast<int>(sizeof(uint8_t)), roi);
  }

  return true;
#else
  NOT_IMPLEMENTED;
  return false;
#endif
}

bool DataProvider::GetImage(const DataProviderImageOptions &options,
                            Image8U *image) {
  AINFO << "GetImage ...";
  if (image == nullptr) {
    return false;
  }
  bool success = false;
  switch (options.target_color) {
    case ImageColorspace::RGB:
      success = to_rgb_image();
      *image = (*rgb_);
      break;
    case ImageColorspace::BGR:
      success = to_bgr_image();
      *image = (*bgr_);
      break;
    case ImageColorspace::GRAY:
      success = to_gray_image();
      *image = (*gray_);
      break;
    default:
      AERROR << "Unsupported Color: "
             << static_cast<uint8_t>(options.target_color);
  }
  if (!success) {
    return false;
  }

  if (options.do_crop) {
    AINFO << "\tcropping ...";
    *image = (*image)(options.crop_roi);
  }
  AINFO << "Done!";
  return true;
}

bool DataProvider::to_gray_image() {
#if USE_GPU == 1
  if (!gray_ready_) {
    NppiSize roi;
    roi.height = src_height_;
    roi.width = src_width_;
    if (bgr_ready_) {
      Npp32f coeffs[] = {0.114f, 0.587f, 0.299f};
      nppiColorToGray_8u_C3C1R(bgr_->gpu_data(), bgr_->width_step(),
                               gray_->mutable_gpu_data(), gray_->width_step(),
                               roi, coeffs);
      gray_ready_ = true;
    } else if (rgb_ready_) {
      Npp32f coeffs[] = {0.299f, 0.587f, 0.114f};
      nppiColorToGray_8u_C3C1R(rgb_->gpu_data(), rgb_->width_step(),
                               gray_->mutable_gpu_data(), gray_->width_step(),
                               roi, coeffs);
      gray_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
#else
  NOT_IMPLEMENTED;
  return false;
#endif
}

bool DataProvider::to_rgb_image() {
#if USE_GPU == 1
  if (!rgb_ready_) {
    NppiSize roi;
    roi.height = src_height_;
    roi.width = src_width_;
    if (bgr_ready_) {
      // BGR2RGB takes less than 0.010ms on K2200
      const int order[] = {2, 1, 0};
      nppiSwapChannels_8u_C3R(bgr_->gpu_data(), bgr_->width_step(),
                              rgb_->mutable_gpu_data(), rgb_->width_step(), roi,
                              order);
      rgb_ready_ = true;
    } else if (gray_ready_) {
      nppiDup_8u_C1C3R(gray_->gpu_data(), gray_->width_step(),
                       rgb_->mutable_gpu_data(), rgb_->width_step(), roi);
      rgb_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
#else
  NOT_IMPLEMENTED;
  return false;
#endif
}

bool DataProvider::to_bgr_image() {
#if USE_GPU == 1
  if (!bgr_ready_) {
    NppiSize roi;
    roi.height = src_height_;
    roi.width = src_width_;
    if (rgb_ready_) {
      const int order[] = {2, 1, 0};
      nppiSwapChannels_8u_C3R(rgb_->gpu_data(), rgb_->width_step(),
                              bgr_->mutable_gpu_data(), bgr_->width_step(), roi,
                              order);
      bgr_ready_ = true;
    } else if (gray_ready_) {
      nppiDup_8u_C1C3R(gray_->gpu_data(), gray_->width_step(),
                       bgr_->mutable_gpu_data(), bgr_->width_step(), roi);
      bgr_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
#else
  NOT_IMPLEMENTED;
  return false;
#endif
}

}  // namespace inference
}  // namespace zetton
