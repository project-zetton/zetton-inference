#include <catch2/catch_test_macros.hpp>

#include "zetton_inference/base/tensor/tensor.h"

using namespace zetton::inference;

TEST_CASE("Test Tensor constructors", "[Tensor]") {
  SECTION("Default constructor") {
    Tensor tensor;
    REQUIRE(tensor.buffer_ == nullptr);
    REQUIRE(tensor.external_data_ptr == nullptr);
    REQUIRE(tensor.dtype == InferenceDataType::kINT8);
    REQUIRE(tensor.shape == std::vector<int64_t>{0});
    REQUIRE(tensor.name == "");
    REQUIRE(tensor.device == InferenceDeviceType::kCPU);
    REQUIRE(tensor.device_id == -1);
    REQUIRE(tensor.is_pinned_memory == false);
    REQUIRE(tensor.temporary_cpu_buffer.empty());
    REQUIRE(tensor.nbytes_allocated == 0);
  }

  SECTION("Constructor with tensor name") {
    Tensor tensor("test_tensor");
    REQUIRE(tensor.name == "test_tensor");
  }

  SECTION("Constructor with scalar") {
    Scalar scalar(1.0f);
    Tensor tensor(scalar);
    REQUIRE(tensor.Numel() == 1);
    REQUIRE(*static_cast<float*>(tensor.MutableData()) == 1.0f);
  }

  SECTION("Deep copy constructor") {
    Tensor tensor1("test_tensor");
    tensor1.Allocate({2, 3}, InferenceDataType::kFP32);
    Tensor tensor2(tensor1);
    REQUIRE(tensor2.name == "test_tensor");
    REQUIRE(tensor2.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor2.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor2.nbytes_allocated == tensor1.nbytes_allocated);
    REQUIRE(tensor2.buffer_ != tensor1.buffer_);
    REQUIRE(std::memcmp(tensor2.MutableData(), tensor1.MutableData(),
                        tensor1.nbytes_allocated) == 0);
  }

  SECTION("Move constructor") {
    Tensor tensor1("test_tensor");
    tensor1.Allocate({2, 3}, InferenceDataType::kFP32);
    void* buffer = tensor1.MutableData();
    Tensor tensor2(std::move(tensor1));
    REQUIRE(tensor2.name == "test_tensor");
    REQUIRE(tensor2.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor2.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor2.nbytes_allocated == 2 * 3 * sizeof(float));
    REQUIRE(tensor2.buffer_ == buffer);
    REQUIRE(tensor1.buffer_ == nullptr);
  }

  SECTION("Deep copy assignment") {
    Tensor tensor1("test_tensor");
    tensor1.Allocate({2, 3}, InferenceDataType::kFP32);
    Tensor tensor2;
    tensor2 = tensor1;
    REQUIRE(tensor2.name == "test_tensor");
    REQUIRE(tensor2.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor2.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor2.nbytes_allocated == tensor1.nbytes_allocated);
    REQUIRE(tensor2.buffer_ != tensor1.buffer_);
    REQUIRE(std::memcmp(tensor2.MutableData(), tensor1.MutableData(),
                        tensor1.nbytes_allocated) == 0);
  }

  SECTION("Move assignment") {
    Tensor tensor1("test_tensor");
    tensor1.Allocate({2, 3}, InferenceDataType::kFP32);
    void* buffer = tensor1.MutableData();
    Tensor tensor2;
    tensor2 = std::move(tensor1);
    REQUIRE(tensor2.name == "test_tensor");
    REQUIRE(tensor2.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor2.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor2.nbytes_allocated == 2 * 3 * sizeof(float));
    REQUIRE(tensor2.buffer_ == buffer);
    REQUIRE(tensor1.buffer_ == nullptr);
  }
}

TEST_CASE("Test Tensor getters", "[Tensor]") {
  Tensor tensor;
  tensor.Allocate({2, 3}, InferenceDataType::kFP32);
  float* data = static_cast<float*>(tensor.MutableData());
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
  data[4] = 5.0f;
  data[5] = 6.0f;

  SECTION("MutableData") {
    float* mutable_data = static_cast<float*>(tensor.MutableData());
    mutable_data[0] = 2.0f;
    mutable_data[1] = 3.0f;
    mutable_data[2] = 4.0f;
    mutable_data[3] = 5.0f;
    mutable_data[4] = 6.0f;
    mutable_data[5] = 7.0f;
    REQUIRE(data[0] == 2.0f);
    REQUIRE(data[1] == 3.0f);
    REQUIRE(data[2] == 4.0f);
    REQUIRE(data[3] == 5.0f);
    REQUIRE(data[4] == 6.0f);
    REQUIRE(data[5] == 7.0f);
  }

  SECTION("Data") {
    const void* const_data = tensor.Data();
    REQUIRE(std::memcmp(const_data, data, tensor.nbytes_allocated) == 0);
  }

  SECTION("CpuData") {
    const void* cpu_data = tensor.CpuData();
    REQUIRE(std::memcmp(cpu_data, data, tensor.nbytes_allocated) == 0);
  }

  SECTION("IsShared") {
    REQUIRE(tensor.IsShared() == false);
    Tensor tensor2;
    tensor2.SetExternalData({2, 3}, InferenceDataType::kFP32, data);
    REQUIRE(tensor2.IsShared() == true);
  }

  SECTION("Nbytes") { REQUIRE(tensor.Nbytes() == 2 * 3 * sizeof(float)); }

  SECTION("Numel") { REQUIRE(tensor.Numel() == 6); }
}

TEST_CASE("Test Tensor setters", "[Tensor]") {
  Tensor tensor;

  SECTION("SetData with copy") {
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor.SetData({2, 3}, InferenceDataType::kFP32, data, true);
    REQUIRE(tensor.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor.nbytes_allocated == 2 * 3 * sizeof(float));
    float* tensor_data = static_cast<float*>(tensor.MutableData());
    REQUIRE(tensor_data[0] == 1.0f);
    REQUIRE(tensor_data[1] == 2.0f);
    REQUIRE(tensor_data[2] == 3.0f);
    REQUIRE(tensor_data[3] == 4.0f);
    REQUIRE(tensor_data[4] == 5.0f);
    REQUIRE(tensor_data[5] == 6.0f);
  }

  SECTION("SetData without copy") {
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor.SetData({2, 3}, InferenceDataType::kFP32, data, false);
    REQUIRE(tensor.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor.nbytes_allocated == 0);
    REQUIRE(tensor.buffer_ == nullptr);
    REQUIRE(tensor.external_data_ptr == data);
  }

  SECTION("SetExternalData") {
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor.SetExternalData({2, 3}, InferenceDataType::kFP32, data);
    REQUIRE(tensor.shape == std::vector<int64_t>{2, 3});
    REQUIRE(tensor.dtype == InferenceDataType::kFP32);
    REQUIRE(tensor.nbytes_allocated == 0);
    REQUIRE(tensor.buffer_ == nullptr);
    REQUIRE(tensor.external_data_ptr == data);
  }
}

TEST_CASE("Test Tensor operations", "[Tensor]") {
  Tensor tensor;
  tensor.Allocate({2, 3}, InferenceDataType::kFP32);
  float* data = static_cast<float*>(tensor.MutableData());
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
  data[4] = 5.0f;
  data[5] = 6.0f;

  SECTION("StopSharing") {
    Tensor tensor2;
    tensor2.SetExternalData({2, 3}, InferenceDataType::kFP32, data);
    tensor2.StopSharing();
    REQUIRE(tensor2.IsShared() == false);
    REQUIRE(tensor2.nbytes_allocated == 2 * 3 * sizeof(float));
    REQUIRE(tensor2.buffer_ != data);
    REQUIRE(std::memcmp(tensor2.MutableData(), data,
                        tensor2.nbytes_allocated) == 0);
  }

  SECTION("Allocate") {
#if USE_GPU == 1
    tensor.Allocate({3, 2}, InferenceDataType::kINT8, "new_tensor",
                    InferenceDeviceType::kGPU);
    REQUIRE(tensor.shape == std::vector<int64_t>{3, 2});
    REQUIRE(tensor.dtype == InferenceDataType::kINT8);
    REQUIRE(tensor.name == "new_tensor");
    REQUIRE(tensor.device == InferenceDeviceType::kGPU);
    REQUIRE(tensor.device_id == -1);
    REQUIRE(tensor.nbytes_allocated == 3 * 2 * sizeof(int8_t));
    REQUIRE(tensor.buffer_ != nullptr);
#else
    tensor.Allocate({3, 2}, InferenceDataType::kINT8, "new_tensor",
                    InferenceDeviceType::kCPU);
    REQUIRE(tensor.shape == std::vector<int64_t>{3, 2});
    REQUIRE(tensor.dtype == InferenceDataType::kINT8);
    REQUIRE(tensor.name == "new_tensor");
    REQUIRE(tensor.device == InferenceDeviceType::kCPU);
    REQUIRE(tensor.device_id == -1);
    REQUIRE(tensor.nbytes_allocated == 3 * 2 * sizeof(int8_t));
    REQUIRE(tensor.buffer_ != nullptr);
#endif
  }

  SECTION("ExpandDim") {
    tensor.ExpandDim(0);
    REQUIRE(tensor.shape == std::vector<int64_t>{1, 2, 3});
    tensor.ExpandDim(2);
    REQUIRE(tensor.shape == std::vector<int64_t>{1, 2, 1, 3});
  }

  SECTION("Resize") {
    tensor.Resize(4 * 3 * sizeof(float));
    REQUIRE(tensor.nbytes_allocated == 4 * 3 * sizeof(float));
#if 0
      tensor.Resize({4, 3});
      REQUIRE(tensor.shape == std::vector<int64_t>{4, 3});
      tensor.Resize({4, 3}, InferenceDataType::kINT8, "new_tensor",
                    InferenceDeviceType::kGPU, 0);
      REQUIRE(tensor.shape == std::vector<int64_t>{4, 3});
      REQUIRE(tensor.dtype == InferenceDataType::kINT8);
      REQUIRE(tensor.name == "new_tensor");
      REQUIRE(tensor.device == InferenceDeviceType::kGPU);
      REQUIRE(tensor.device_id == 0);
      REQUIRE(tensor.nbytes_allocated == 4 * 3 * sizeof(int8_t));
      REQUIRE(tensor.buffer_ != nullptr);
#endif
  }
}
