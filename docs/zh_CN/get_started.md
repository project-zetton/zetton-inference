---
title: 快速入门指南
---

- [快速入门指南](#快速入门指南)
  - [预置条件](#预置条件)
  - [获取并编译 zetton-inference 代码](#获取并编译-zetton-inference-代码)
  - [在其他项目中使用 zetton-common](#在其他项目中使用-zetton-common)

# 快速入门指南

这份文档旨在协助您使用 CMake 搭建 zetton-inference 的开发环境。我们强烈建议每个想要开始使用 zetton-inference 进行代码开发的人都至少完成一遍这个快速教程，以便更好地掌握相关技能。

## 预置条件

在本教程中，编译并运行 zetton-common 代码需要：

1. [zetton-common](https://github.com/project-zetton/zetton-common) 及其依赖的软件环境

2. [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page): 用于矩阵运算

3. [OpenCV](https://opencv.org/): 用于图像处理

上述所有依赖库，可以参考 [zetton-docker](https://github.com/project-zetton/zetton-docker) 中的 `Dockerfile` 进行安装。或者直接使用 zetton-docker 提供的 Docker 容器来运行 zetton-common 代码。

## 获取并编译 zetton-inference 代码

支持使用 CMake 和 colcon 进行构建。推荐使用 colcon 进行构建。

具体的构建方法可参照 [zetton-common] 的快速入门指南中的对应部分。

## 在其他项目中使用 zetton-common

支持在其他的 CMake 项目或者 colcon 工作空间中使用 zetton-common。具体的使用方法可参照 [zetton-common] 的快速入门指南中的对应部分。
