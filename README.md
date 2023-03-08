# zetton-inference

English | [中文](README_zh-CN.md)

## Table of Contents

- [zetton-inference](#zetton-inference)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [What's New](#whats-new)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Overview of Benchmark and Model Zoo](#overview-of-benchmark-and-model-zoo)
  - [FAQ](#faq)
  - [Contributing](#contributing)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)
  - [Related Projects](#related-projects)

## Introduction

zetton-inference is an open source package for deep learning inference. It's a part of the [Project Zetton](https://github.com/project-zetton).

<details open>
<summary>Major features</summary>

- **Modular Design**: zetton-inference is designed to be modular, which means that you can easily add new inference nodes to the package.

- **Support Multiple Frameworks**: zetton-inference supports multiple deep learning frameworks, such as ONNX, TensorRT, RKNN, OpenVINO, etc.

- **High Efficiency**: zetton-inference is designed to be high efficient, which means that you can easily deploy the inference nodes to GPU servers or embedded devices.

- **State-of-the-art Algorithms**: zetton-inference provides state-of-the-art algorithms, such as object detection, object tracking, etc.

</details>

## What's New

- (2022-09-19) TensorRT-based inference nodes are moved to [zetton-inference-tensorrt](https://github.com/project-zetton/zetton-inference-tensorrt)
- (2022-10-08) this repo is conevrted into a pure CMake package.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

For compatibility changes between different versions of zetton-inference, please refer to [compatibility.md](docs/en/compatibility.md).

## Installation

Please refer to [Installation](docs/en/get_started.md) for installation instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of zetton-inference.

## Overview of Benchmark and Model Zoo

| Task      | Model     | ONNX | TensorRT | RKNN | OpenVINO |
| :-------- | :-------- | :--- | :------- | :--- | :------- |
| Detection | YOLOv5    | ✅    | ✅        | ❌    | ❌        |
| Detection | YOLOX     | ✅    | ✅        | ❌    | ❌        |
| Detection | YOLOv7    | ✅    | ✅        | ❌    | ❌        |
| Tracking  | DeepSORT  | /    | /        | /    | /        |
| Tracking  | ByteTrack | /    | /        | /    | /        |

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve zetton-inferenece. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the package and benchmark could serve the growing research and production community by providing a flexible toolkit to deploy models.

## License

- For academic use, this project is licensed under the 2-clause BSD License, please see the [LICENSE file](LICENSE) for details.
- For commercial use, please contact [Yusu Pan](mailto:xxdsox@gmail.com).

## Related Projects

- [zetton-inference-tensorrt](https://github.com/project-zetton/zetton-inference-tensorrt): TensorRT-based inference nodes for Project Zetton.

- [zetton-ros-vendor](https://github.com/project-zetton/zetton-ros-vendor):
ROS-related examples are moved to this package.
