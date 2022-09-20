#pragma once

#include <cstdint>
#include <cwchar>
#include <opencv2/core/mat.hpp>

#include "zetton_common/time/time.h"
#include "zetton_inference/base/frame/frame.h"

namespace zetton {
namespace inference {

CameraFramePtr CreateCameraFrame(const int& image_width,
                                 const int& image_height) {
  // init data provider
  DataProviderInitOptions data_provider_init_options;
  data_provider_init_options.image_height = image_height;
  data_provider_init_options.image_width = image_width;
  std::shared_ptr<DataProvider> data_provider(new DataProvider);
  data_provider->Init(data_provider_init_options);

  // init camera frame
  CameraFramePtr frame(new CameraFrame);
  frame->data_provider = data_provider.get();

  return frame;
}

bool FeedCvMatToCameraFrame(const cv::Mat& cv_mat, CameraFramePtr& frame,
                            const double& frame_timestamp = -1.0,
                            const std::string& frame_encoding = "bgr8") {
  // set frame id
  frame->frame_id += 1;

  // set frame timestamp
  if (frame_timestamp < 0) {
    frame->timestamp = common::Time::Now().ToSecond();
  } else {
    frame->timestamp = frame_timestamp;
  }

  // set frame data provider
  frame->data_provider->FillImageData(
      cv_mat.rows, cv_mat.cols, reinterpret_cast<const uint8_t*>(cv_mat.data),
      frame_encoding);

  return true;
}

bool FeedRawDataToCameraFrame(const uint8_t* data, CameraFramePtr& frame,
                              const double& frame_timestamp = -1.0,
                              const std::string& frame_encoding = "bgr8") {
  // set frame id
  frame->frame_id += 1;

  // set frame timestamp
  if (frame_timestamp < 0) {
    frame->timestamp = common::Time::Now().ToSecond();
  } else {
    frame->timestamp = frame_timestamp;
  }

  // set frame data provider
  frame->data_provider->FillImageData(frame->data_provider->src_height(),
                                      frame->data_provider->src_width(), data,
                                      frame_encoding);

  return true;
}

CameraFramePtr CreateCameraFrameFromCvMat(
    const cv::Mat& cv_mat, const double& frame_timestamp = -1.0,
    const std::string& frame_encoding = "bgr8") {
  auto frame = CreateCameraFrame(cv_mat.cols, cv_mat.rows);
  FeedCvMatToCameraFrame(cv_mat, frame, frame_timestamp, frame_encoding);
  return frame;
}

}  // namespace inference
}  // namespace zetton