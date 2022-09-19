#pragma once

#include <string>

#include "zetton_common/util/registerer.h"
#include "zetton_inference/base/geometry/box.h"

namespace zetton {
namespace inference {

struct DataProviderInitOptions {
  int image_height = 0;
  int image_width = 0;
  int device_id = -1;
};

class BaseDataProvider {
 public:
  BaseDataProvider() = default;
  ~BaseDataProvider() = default;

  BaseDataProvider(const BaseDataProvider &) = delete;
  BaseDataProvider &operator=(const BaseDataProvider &) = delete;

 public:
  virtual bool Init(
      const DataProviderInitOptions &init_options = DataProviderInitOptions()) {
    src_height_ = init_options.image_height;
    src_width_ = init_options.image_width;
    device_id_ = init_options.device_id;

    return true;
  };
  virtual bool FillImageData(int rows, int cols, const uint8_t *data,
                             const std::string &encoding) = 0;

 public:
  int src_height() const { return src_height_; }
  int src_width() const { return src_width_; }

 protected:
  int src_height_ = 0;
  int src_width_ = 0;
  int device_id_ = -1;
};

ZETTON_REGISTER_REGISTERER(BaseDataProvider)
#define ZETTON_REGISTER_DATA_PROVIDER(name) \
  ZETTON_REGISTER_CLASS(BaseDataProvider, name)

}  // namespace inference
}  // namespace zetton
