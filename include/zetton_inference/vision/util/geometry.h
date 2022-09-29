#pragma once

#include <array>
#include <cmath>

namespace zetton {
namespace inference {
namespace vision {

/// \brief convert bounding box from [cx,cy,a,h] to [x1,y1,x2,y2] style
inline std::array<float, 4> GetXYAHFromTLBR(const std::array<float, 4>& tlbr) {
  return {
      (tlbr[0] + tlbr[2]) / 2,
      (tlbr[1] + tlbr[3]) / 2,
      (tlbr[2] - tlbr[0]) / (tlbr[3] - tlbr[1]),
      (tlbr[3] - tlbr[1]),
  };
}

/// \brief convert bounding box from [x1,y1,x2,y2] to [cx,cy,a,h] style
inline std::array<float, 4> GetTLBRFromXYAH(const std::array<float, 4>& xyah) {
  return {
      xyah[0] - xyah[2] * xyah[3] / 2,
      xyah[1] - xyah[3] / 2,
      xyah[0] + xyah[2] * xyah[3] / 2,
      xyah[1] + xyah[3] / 2,
  };
}

/// \brief convert bounding box from [cx,cy,a,h] to [x,y,w,h] style
inline std::array<float, 4> GetXYAHFromTLWH(const std::array<float, 4>& tlwh) {
  return {
      tlwh[0] + tlwh[2] / 2,
      tlwh[1] + tlwh[3] / 2,
      tlwh[2] / tlwh[3],
      tlwh[3],
  };
}

/// \brief convert bounding box from [x,y,w,h] to [cx,cy,a,h] style
inline std::array<float, 4> GetTLWHFromXYAH(const std::array<float, 4>& xyah) {
  return {
      xyah[0] - xyah[2] * xyah[3] / 2,
      xyah[1] - xyah[3] / 2,
      xyah[2] * xyah[3],
      xyah[3],
  };
}

/// \brief convert bounding box from [x1,y1,x2,y2] to [x,y,w,h] style
inline std::array<float, 4> GetTLBRFromTLWH(const std::array<float, 4>& tlwh) {
  return {
      tlwh[0],
      tlwh[1],
      tlwh[0] + tlwh[2],
      tlwh[1] + tlwh[3],
  };
}

/// \brief convert bounding box from [x,y,w,h] to [x1,y1,x2,y2] style
inline std::array<float, 4> GetTLWHFromTLBR(const std::array<float, 4>& tlbr) {
  return {
      tlbr[0],
      tlbr[1],
      tlbr[2] - tlbr[0],
      tlbr[3] - tlbr[1],
  };
}

/// \brief convert bounding box from [x1,y1,x2,y2] to [cx,cy,s,r] style
inline std::array<float, 4> GetXYSRFromTLBR(const std::array<float, 4>& tlbr) {
  return {
      (tlbr[0] + tlbr[2]) / 2,
      (tlbr[1] + tlbr[3]) / 2,
      (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1]),
      (tlbr[2] - tlbr[0]) / (tlbr[3] - tlbr[1]),
  };
}

/// \brief convert bounding box from [cx,cy,s,r] to [x1,y1,x2,y2] style
inline std::array<float, 4> GetTLBRFromXYSR(const std::array<float, 4>& xysr) {
  auto w = std::sqrt(xysr[2] * xysr[3]);
  auto h = xysr[2] / w;
  return {
      xysr[0] - w / 2,
      xysr[1] - h / 2,
      xysr[0] + w / 2,
      xysr[1] + h / 2,
  };
}

/// \brief convert bounding box from [cx,cy,s,r] to [x,y,w,h] style
inline std::array<float, 4> GetTLWHFromXYSR(const std::array<float, 4>& xysr) {
  auto w = std::sqrt(xysr[2] * xysr[3]);
  auto h = xysr[2] / w;
  return {
      xysr[0] - w / 2,
      xysr[1] - h / 2,
      w,
      h,
  };
}

/// \brief convert bounding box from [x,y,w,h] to [cx,cy,s,r] style
inline std::array<float, 4> GetTLSRFromTLWH(const std::array<float, 4>& tlwh) {
  return {
      tlwh[0] + tlwh[2] / 2,
      tlwh[1] + tlwh[3] / 2,
      tlwh[2] * tlwh[3],
      tlwh[2] / tlwh[3],
  };
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
