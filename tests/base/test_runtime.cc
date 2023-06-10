
#include <catch2/catch_test_macros.hpp>

#include "zetton_inference/base/runtime.h"
#include "zetton_inference/util/runtime_util.h"

using namespace zetton::inference;

TEST_CASE("Test InferenceRuntime initialization", "[InferenceRuntime]") {
  SECTION("Initialization with empty options") {
    InferenceRuntime runtime;
    auto options = std::make_shared<InferenceRuntimeOptions>();
    REQUIRE_FALSE(runtime.Init(options.get()));
  }

  SECTION("Initialization with valid options") {
    InferenceRuntime runtime;
    auto options = std::make_shared<InferenceRuntimeOptions>();
    options->model_format = InferenceFrontendType::kONNX;
    options->device = InferenceDeviceType::kCPU;
    options->backend = InferenceBackendType::kONNXRuntime;
    if (IsBackendAvailable(InferenceBackendType::kONNXRuntime)) {
      REQUIRE(runtime.Init(options.get()));
    }
  }
}

TEST_CASE("Test InferenceRuntime inference", "[InferenceRuntime]") {
  SECTION("Inference with given input and output") {}

  SECTION("Inference with binded input and output") {}
}

TEST_CASE("Test InferenceRuntime getters", "[InferenceRuntime]") {
  SECTION("Get the number of input tensors") {}

  SECTION("Get the number of output tensors") {}

  SECTION("Get the input tensor info by index") {}

  SECTION("Get the output tensor info by index") {}

  SECTION("Get all the input tensor infos") {}

  SECTION("Get all the output tensor infos") {}

  SECTION("Get the output tensor by name") {}
}

TEST_CASE("Test InferenceRuntime setters", "[InferenceRuntime]") {
  SECTION("Bind input tensor") {}

  SECTION("Bind output tensor") {}
}

TEST_CASE("Test InferenceRuntime operations", "[InferenceRuntime]") {
  SECTION("Clone the InferenceRuntime") {}
}
