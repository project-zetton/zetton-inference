#pragma once

#include <zetton_common/log/log.h>

#include <cstdint>
#include <limits>

#include "zetton_inference/base/type.h"

#define ENABLE_NAIVE_FP16 0

namespace zetton {
namespace inference {

class Scalar {
 public:
  // Constructor support implicit
  Scalar() : Scalar(0) {}
  Scalar(double val) : dtype_(InferenceDataType::kFP64) { data_.f64 = val; }

  Scalar(float val) : dtype_(InferenceDataType::kFP32) { data_.f32 = val; }

#if ENABLE_NAIVE_FP16
  // FIXME: naive float16 is not supported in C++ standard yet
  Scalar(float16 val) : dtype_(InferenceDataType::kFP16) { data_.f16 = val; }
#endif

  Scalar(int64_t val) : dtype_(InferenceDataType::kINT64) { data_.i64 = val; }

  Scalar(int32_t val) : dtype_(InferenceDataType::kINT32) { data_.i32 = val; }

  Scalar(int16_t val) : dtype_(InferenceDataType::kINT16) { data_.i16 = val; }

  Scalar(int8_t val) : dtype_(InferenceDataType::kINT8) { data_.i8 = val; }

  Scalar(uint8_t val) : dtype_(InferenceDataType::kUINT8) { data_.ui8 = val; }

  Scalar(bool val) : dtype_(InferenceDataType::kBOOL) { data_.b = val; }

  // The compatible method for fliud operators,
  // and it will be removed in the future.
  explicit Scalar(const std::string& str_value)
      : dtype_(InferenceDataType::kFP64) {
    if (str_value == "inf") {
      data_.f64 = std::numeric_limits<double>::infinity();
    } else if (str_value == "-inf") {
      data_.f64 = -std::numeric_limits<double>::infinity();
    } else if (str_value == "nan") {
      data_.f64 = std::numeric_limits<double>::quiet_NaN();
    } else {
      data_.f64 = std::stod(str_value);
    }
  }

  template <typename RT>
  inline RT to() const {
    switch (dtype_) {
      case InferenceDataType::kFP32:
        return static_cast<RT>(data_.f32);
      case InferenceDataType::kFP64:
        return static_cast<RT>(data_.f64);
#if ENABLE_NAIVE_FP16
      case InferenceDataType::kFP16:
        return static_cast<RT>(data_.f16);
#endif
      case InferenceDataType::kINT32:
        return static_cast<RT>(data_.i32);
      case InferenceDataType::kINT64:
        return static_cast<RT>(data_.i64);
      case InferenceDataType::kINT16:
        return static_cast<RT>(data_.i16);
      case InferenceDataType::kINT8:
        return static_cast<RT>(data_.i8);
      case InferenceDataType::kUINT8:
        return static_cast<RT>(data_.ui8);
      case InferenceDataType::kBOOL:
        return static_cast<RT>(data_.b);
      default:
        ACHECK_F(false, "Invalid enum scalar data type `{}`.",
                 ToString(dtype_));
    }
  }

  InferenceDataType dtype() const { return dtype_; }

 private:
  InferenceDataType dtype_;
  union data {
    bool b;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t ui8;
#if ENABLE_NAIVE_FP16
    float16 f16;
#endif
    float f32;
    double f64;
  } data_;
};

}  // namespace inference
}  // namespace zetton
