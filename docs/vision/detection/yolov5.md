# YOLOv5 models

Supported backends:

- TensorRT

## YOLOv5-End2End

Model conversion:

```bash
# Download YOLOv5 code
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
# Download trained weights
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
# Export temporary ONNX model for TensorRT converter
python export.py --weights yolov5n.pt --include onnx --simplify --inplace
# Download ONNX to TensorRT converter
cd ../
git clone https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git
# Export TensorRT-engine model
cd TensorRT-For-YOLO-Series
python export.py -o ../yolov5/yolov5n.onnx -e yolov5n-nms.trt --end2end
```

Model inferenece: `zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel`

Example: `zetton-inference-tensorrt/examples/run_yolov7_end2end.cc`

## Acknowledgements

- [Offical PyTorch Implementation of YOLOv5](https://github.com/ultralytics/yolov5)
- [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git)
