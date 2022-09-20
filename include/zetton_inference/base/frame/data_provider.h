#pragma once

#include <string>

#include "zetton_inference/base/frame/image.h"

namespace zetton {
namespace inference {

struct DataProviderInitOptions {
  DataProviderInitOptions()
      : image_height(0),
        image_width(0),
        device_id(-1),
        do_undistortion(false) {}

  int image_height;
  int image_width;
  int device_id;
  bool do_undistortion;
  std::string sensor_name;
};

struct DataProviderImageOptions {
  DataProviderImageOptions() {
    this->target_color = ImageColorspace::NONE;
    this->do_crop = false;
  }

  DataProviderImageOptions(ImageColorspace target_color, bool do_crop,
                           RectI crop_roi) {
    this->target_color = target_color;
    this->do_crop = do_crop;
    this->crop_roi = crop_roi;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << " " << static_cast<int>(target_color);
    ss << " " << do_crop;
    if (do_crop) {
      ss << " " << crop_roi.x << " " << crop_roi.y << " " << crop_roi.width
         << " " << crop_roi.height;
    }
    return ss.str();
  }

  ImageColorspace target_color = ImageColorspace::BGR;
  bool do_crop = false;
  RectI crop_roi;
};

class DataProvider {
 public:
  DataProvider() = default;
  ~DataProvider() = default;

  DataProvider(const DataProvider &) = delete;
  DataProvider &operator=(const DataProvider &) = delete;

  bool Init(const DataProviderInitOptions &options = DataProviderInitOptions());

  /// \brief fill raw image data.
  /// \param options
  /// \param blob image blob with specified size should be filled, required.
  bool FillImageData(int rows, int cols, const uint8_t *data,
                     const std::string &encoding);

  /// \brief get blob converted from raw message.
  /// \param options
  /// \param blob (4D,NHWC) image blob with specified size should be filled,
  /// required.
  bool GetImageBlob(const DataProviderImageOptions &options,
                    Blob<uint8_t> *blob);

  /// \brief get Image8U converted from raw message.
  /// \param options
  bool GetImage(const DataProviderImageOptions &options, Image8U *image);

  int src_height() const { return src_height_; }
  int src_width() const { return src_width_; }
  const std::string &sensor_name() const { return sensor_name_; }

  bool to_gray_image();
  bool to_rgb_image();
  bool to_bgr_image();

 protected:
  std::string sensor_name_;
  int src_height_ = 0;
  int src_width_ = 0;
  int device_id_ = -1;

  std::shared_ptr<Image8U> gray_;
  std::shared_ptr<Image8U> rgb_;
  std::shared_ptr<Image8U> bgr_;
  bool gray_ready_ = false;
  bool rgb_ready_ = false;
  bool bgr_ready_ = false;

  Blob<float> temp_float_;
  Blob<uint8_t> temp_uint8_;
};

}  // namespace inference
}  // namespace zetton
