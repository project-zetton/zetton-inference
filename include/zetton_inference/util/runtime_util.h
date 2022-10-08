#pragma once

#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_backend.h"

namespace zetton {
namespace inference {

/// \brief Get all available inference backend types
std::vector<InferenceBackendType> GetAvailableBackends();

/// \brief Check if the specified inference backend is available
/// \param backend_type inference backend type
bool IsBackendAvailable(const InferenceBackendType& backend);

/// \brief Check if the model file is related to the specified inference
/// frontend type
/// \param model_file model file path
/// \param frontend_type inference frontend type
bool CheckModelFormat(const std::string& model_file,
                      const InferenceFrontendType& model_format);

/// \brief Guess the inference frontend type of the specified model file
/// \param model_file model file path
InferenceFrontendType GuessModelFormat(const std::string& model_file);

}  // namespace inference
}  // namespace zetton
