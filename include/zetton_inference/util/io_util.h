#pragma once

#include <string>

namespace zetton {
namespace inference {

bool ReadBinaryFromFile(const std::string& file, std::string* contents);

}
}  // namespace zetton
