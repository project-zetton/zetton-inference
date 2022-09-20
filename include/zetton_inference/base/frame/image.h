#pragma once

#include <map>
#include <memory>
#include <utility>

#include "zetton_common/log/log.h"
#include "zetton_inference/base/frame/blob.h"
#include "zetton_inference/base/geometry/box.h"

namespace zetton {
namespace inference {

enum class ImageColorspace {
  NONE = 0x00,
  GRAY = 0x01,
  RGB = 0x02,
  BGR = 0x03,
};

const std::map<ImageColorspace, int> kChannelsMap{{ImageColorspace::GRAY, 1},
                                                  {ImageColorspace::RGB, 3},
                                                  {ImageColorspace::BGR, 3}};

/// \brief A wrapper around Blob holders serving as the basic computational unit
/// for images.
class Image8U {
 public:
  Image8U()
      : rows_(0),
        cols_(0),
        type_(ImageColorspace::NONE),
        channels_(0),
        width_step_(0),
        blob_(nullptr),
        offset_(0) {}

  Image8U(int rows, int cols, ImageColorspace type,
          std::shared_ptr<Blob<uint8_t>> blob, int offset = 0)
      : rows_(rows),
        cols_(cols),
        type_(type),
        blob_(std::move(blob)),
        offset_(offset) {
    channels_ = kChannelsMap.at(type);
    ACHECK_EQ(blob_->num_axes(), 3);
    ACHECK_EQ(blob_->shape(2), channels_);
    ACHECK_LE(offset_ + blob_->offset({rows - 1, cols - 1, channels_ - 1}),
              (int)(blob_->count()));
    width_step_ = blob_->offset({1, 0, 0}) * static_cast<int>(sizeof(uint8_t));
  }

  Image8U(int rows, int cols, ImageColorspace type)
      : rows_(rows), cols_(cols), type_(type), offset_(0) {
    channels_ = kChannelsMap.at(type);
    blob_.reset(new Blob<uint8_t>({rows_, cols_, channels_}));
    width_step_ = blob_->offset({1, 0, 0}) * static_cast<int>(sizeof(uint8_t));
  }

  Image8U(const Image8U &src)
      : rows_(src.rows_),
        cols_(src.cols_),
        type_(src.type_),
        channels_(src.channels_),
        width_step_(src.width_step_),
        blob_(src.blob_),
        offset_(src.offset_) {}

  Image8U &operator=(const Image8U &src) {
    this->rows_ = src.rows_;
    this->cols_ = src.cols_;
    this->type_ = src.type_;
    this->channels_ = src.channels_;
    this->width_step_ = src.width_step_;
    this->blob_ = src.blob_;
    this->offset_ = src.offset_;
    return *this;
  }

  ~Image8U() {}

  uint8_t *mutable_cpu_data() { return mutable_cpu_ptr(0); }

  uint8_t *mutable_gpu_data() { return mutable_gpu_ptr(0); }

  const uint8_t *cpu_data() const { return cpu_ptr(0); }

  const uint8_t *gpu_data() const { return gpu_ptr(0); }

  const uint8_t *cpu_ptr(int row = 0) const {
    return blob_->cpu_data() + blob_->offset({row, 0, 0}) + offset_;
  }

  const uint8_t *gpu_ptr(int row = 0) const {
    return blob_->gpu_data() + blob_->offset({row, 0, 0}) + offset_;
  }

  uint8_t *mutable_cpu_ptr(int row = 0) {
    return blob_->mutable_cpu_data() + blob_->offset({row, 0, 0}) + offset_;
  }

  uint8_t *mutable_gpu_ptr(int row = 0) {
    return blob_->mutable_gpu_data() + blob_->offset({row, 0, 0}) + offset_;
  }

  ImageColorspace type() const { return type_; }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int channels() const { return channels_; }
  int width_step() const { return width_step_; }
  /// \brief returns the total number of pixels.
  int total() const { return rows_ * cols_ * channels_; }

  Image8U operator()(const Rect<int> &roi) {
    int offset = offset_ + blob_->offset({roi.y, roi.x, 0});
    // return Image8U(roi.height, roi.width, type_, blob_, offset);
    return Image8U(roi.height, roi.width, type_, blob_, offset);
  }

  std::shared_ptr<Blob<uint8_t>> blob() { return blob_; }

  // DONOT return `std::shared_ptr<const Blob<uint8_t>> &` or `const std::... &`
  std::shared_ptr<const Blob<uint8_t>> blob() const { return blob_; }

 protected:
  int rows_;
  int cols_;
  ImageColorspace type_;
  int channels_;
  int width_step_;
  std::shared_ptr<Blob<uint8_t>> blob_;
  int offset_;
};

using Image8UPtr = std::shared_ptr<Image8U>;
using Image8UConstPtr = std::shared_ptr<const Image8U>;

}  // namespace inference
}  // namespace zetton
