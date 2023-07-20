#include "zetton_inference/util/io.h"

#include <fstream>

#include "zetton_common/log/log.h"

namespace zetton {
namespace inference {

bool ReadBinaryFromFile(const std::string& file, std::string* contents) {
  std::ifstream fin(file, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    AERROR_F("Failed to open file: {}", file);
    return false;
  }
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
  return true;
}

}  // namespace inference
}  // namespace zetton
