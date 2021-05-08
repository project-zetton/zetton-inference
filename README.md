# zetton-inference

Deep learning inference nodes in ROS enviroment with support for both PC and Jetson devices.

## Prerequisites

Recommended PC environment:

- Ubuntu 18.04
- ROS Melodic
- OpenCV 4.1.0+
- GStreamer 1.14+
- CUDA 10.0+
- cuDNN 7.6.5+
- TensorRT 7.0.0+

Recommended Jetson environment:

- JetPack 4.4+ w/ all above packages installed

## Usage

### Object Detection

#### YOLO

Object detection powered by YOLO-family algorithms.

- Receive image form ROS topic and do detection:

   ```bash
   rosrun zetton_inference example_ros_yolo_object_detector
   ```

- Receive image form RTSP stream and do detection:

   ```bash
   rosrun zetton_inference example_rtsp_yolo_object_detector
   ```

### Object Tracking

#### SORT

Object tracking powered by SORT algorithms.

- Receive image form ROS topic, and then do detection & tracking:

   ```bash
   rosrun zetton_inference example_ros_mot_tracker
   ```

#### MOT

Object tracking powered by Optical Flow & ReID.

(W.I.P)

## License

- For academic use, this project is licensed under the 2-clause BSD License - see the [LICENSE file](LICENSE) for details.
- For commercial use, please contact [Yusu Pan](mailto:xxdsox@gmail.com).
