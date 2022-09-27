# YOLOv7 models

Supported backends:

- TensorRT

## YOLOv7-End2End

Model conversion:

```bash
# Download YOLOv7 code
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
# Download trained weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# Export temporary ONNX model for TensorRT converter
python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
# Download ONNX to TensorRT converter
cd ../
git clone https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git
# Export TensorRT-engine model
cd TensorRT-For-YOLO-Series
python3 export.py -o /content/yolov7/yolov7-tiny.onnx -e ./yolov7-tiny-nms.trt -p fp16
```

Model inferenece: `zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel`

Example: `zetton-inference-tensorrt/examples/run_yolov7_end2end.cc`

## Acknowledgements

- [Offical PyTorch Implementation of YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git)
