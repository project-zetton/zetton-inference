---
title: 代码地图
---

- [代码地图](#代码地图)
  - [目录结构](#目录结构)
  - [模块介绍](#模块介绍)

# 代码地图

本文档旨在帮助您了解 zetton-common 代码的结构和功能。

## 目录结构

zetton-inference 代码的目录结构如下：

```bash
$ tree -L 3

.
├── assets/
├── cmake/
├── CMakeLists.txt
├── docs/
├── examples/
├── include/
├── LICENSE
├── package.xml
├── README.md
├── src/
└── tools/
    ├── githooks/
    │   └── commit-msg*
    └── install_git_hooks.sh*
```

其中：

- `cmake/` 与 `CMakeLists.txt`：CMake 构建相关的文件
- `docs/`：文档目录
- `examples/`：示例代码目录
- `include/`：头文件目录
- `src/`：源代码目录
- `tools/`：工具脚本目录
- `LICENSE`：软件包许可证
- `README.md`：软件包说明文档
- `package.xml`：软件包描述文件，用于 colcon 构建

## 模块介绍

zetton-inference 代码包含如下模块：

- `base`：基础模块，包含一些基础的数据结构和工具函数
- `util`：工具模块，包含一些常用的工具函数
- `vision`：视觉模块，包含一些视觉相关的模型和算法
