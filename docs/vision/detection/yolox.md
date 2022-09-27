# YOLOX models

Supported backends:

- TensorRT

## YOLOX-End2End

Model conversion:

```bash
# Download YOLOX code
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd yolov7
# Install dependencies
python3 -m pip install loguru tabulate
# Download trained weights
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
# Export temporary ONNX model for TensorRT converter
python tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth --decode_in_inference
# Download ONNX to TensorRT converter
cd ../
git clone https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git
# Export TensorRT-engine model
cd TensorRT-For-YOLO-Series
python3 export.py -o ../YOLOX/yolox_s.onnx -e yolox_s-nms.trt -p fp16
```

Model inferenece: `zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel`

Example: `zetton-inference-tensorrt/examples/run_yolov7_end2end.cc`

Notes:

- YOLOX models adopt BGR images in `[0,255]` by default, according to [this issue](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/11), so we need to disable these transforms in preprocessing. Remember to explicitly declare the model type in the `Predict` method.

   ```cpp
   detector->Predict(&image, &result, 0.25,
                     zetton::inference::vision::YOLOEnd2EndModelType::kYOLOX);
   ```

## Acknowledgements

- [Offical PyTorch Implementation of YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git)
