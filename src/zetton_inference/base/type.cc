#include "zetton_inference/base/type.h"

#include "zetton_common/log/log.h"

namespace zetton {
namespace inference {

std::string ToString(const InferenceDeviceType& device) {
  if (device == InferenceDeviceType::kCPU) {
    return "CPU";
  } else if (device == InferenceDeviceType::kGPU) {
    return "GPU";
  } else {
    return "Unknown";
  }
}

std::string ToString(const InferenceDataType& dtype) {
  if (dtype == InferenceDataType::kBOOL) {
    return "BOOL";
  } else if (dtype == InferenceDataType::kUINT8) {
    return "UINT8";
  } else if (dtype == InferenceDataType::kINT8) {
    return "INT8";
  } else if (dtype == InferenceDataType::kINT16) {
    return "INT16";
  } else if (dtype == InferenceDataType::kINT32) {
    return "INT32";
  } else if (dtype == InferenceDataType::kINT64) {
    return "INT64";
  } else if (dtype == InferenceDataType::kFP16) {
    return "FP16";
  } else if (dtype == InferenceDataType::kFP32) {
    return "FP32";
  } else if (dtype == InferenceDataType::kFP64) {
    return "FP64";
  } else {
    return "Unknown";
  }
}

std::string ToString(const InferenceBackendType& backend) {
  if (backend == InferenceBackendType::kTensorRT) {
    return "TensorRT";
  } else if (backend == InferenceBackendType::kONNXRuntime) {
    return "ONNXRuntime";
  } else if (backend == InferenceBackendType::kNCNN) {
    return "NCNN";
  } else if (backend == InferenceBackendType::kOpenVINO) {
    return "OpenVINO";
  } else if (backend == InferenceBackendType::kRKNN) {
    return "RKNN";
  } else {
    return "Unknown";
  }
}

std::string ToString(const InferenceFrontendType& frontend) {
  if (frontend == InferenceFrontendType::kONNX) {
    return "ONNX";
  } else if (frontend == InferenceFrontendType::kSerialized) {
    return "Serialized";
  } else if (frontend == InferenceFrontendType::kAuto) {
    return "Auto";
  } else {
    return "Unknown";
  }
}

int32_t InferenceDataTypeSize(const InferenceDataType& dtype) {
  if (dtype == InferenceDataType::kBOOL) {
    return sizeof(bool);
  } else if (dtype == InferenceDataType::kINT16) {
    return sizeof(int16_t);
  } else if (dtype == InferenceDataType::kINT32) {
    return sizeof(int32_t);
  } else if (dtype == InferenceDataType::kINT64) {
    return sizeof(int64_t);
  } else if (dtype == InferenceDataType::kFP16) {
    return sizeof(float) / 2;
  } else if (dtype == InferenceDataType::kFP32) {
    return sizeof(float);
  } else if (dtype == InferenceDataType::kFP64) {
    return sizeof(double);
  } else if (dtype == InferenceDataType::kUINT8) {
    return sizeof(uint8_t);
  } else if (dtype == InferenceDataType::kINT8) {
    return sizeof(int8_t);
  } else {
    AFATAL_F("Unexpected data type: {}", ToString(dtype));
    return -1;
  }
}

}  // namespace inference
}  // namespace zetton
