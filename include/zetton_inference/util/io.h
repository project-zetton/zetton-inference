#pragma once

#include <string>

namespace zetton {
namespace inference {

/// \brief read file content to string
/// \param file_path file path
/// \param content file content
/// \return true if read successfully, otherwise false
bool ReadBinaryFromFile(const std::string& file, std::string* contents);

}  // namespace inference
}  // namespace zetton
