# YOLOv7 in TensorRT backend

Supported backends:

- TensorRT

## YOLOv7-End2End

```bash
# Download YOLOv7 code
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
# Download trained weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# Export temporary ONNX model for TensorRT converter
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
# Download ONNX to TensorRT converter
cd ../
git clone https://github.com/Linaom1214/tensorrt-python.git
# Export TensorRT-engine model
python export.py -o /content/yolov7/yolov7-tiny.onnx -e ./yolov7-tiny-nms.trt -p fp16
```

## Acknowledgements

- [Offical PyTorch Implementation](https://github.com/WongKinYiu/yolov7)
