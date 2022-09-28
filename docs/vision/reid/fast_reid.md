# FastReID models

Supported backends:

- TensorRT

Model conversion:

```bash
# Download FastReID code
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
# Download trained weights
wget https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth
# Export temporary ONNX model for TensorRT converter
python3 -m pip install -U pip
python3 -m pip install onnxoptimizer==0.2.7 yacs
python3 tools/deploy/onnx_export.py --config-file configs/Market1501/bagtricks_R50.yml --name baseline_R50 --output outputs/onnx_model --opts MODEL.WEIGHTS market_bot_R50.pth
# Export TensorRT-engine model
python3 tools/deploy/trt_export.py --name baseline_R50 --output outputs/trt_model --mode fp32 --batch-size 8 --height 256 --width 128 --onnx-model outputs/onnx_model/baseline_R50.onnx
```

Patch for `tools/deploy/trt_export.py` if TensorRT version > 7:

```diff
--- a/tools/deploy/trt_export.py
+++ b/tools/deploy/trt_export.py
@@ -142,7 +142,10 @@ def onnx2trt(
         config.set_flag(trt.BuilderFlag.STRICT_TYPES)

     logger.info("Building an engine from file {}; this may take a while...".format(onnx_file_path))
-    engine = builder.build_cuda_engine(network)
+    # engine = builder.build_cuda_engine(network)
+    plan = builder.build_serialized_network(network, config)
+    with trt.Runtime(trt_logger) as runtime:
+        engine = runtime.deserialize_cuda_engine(plan)
     logger.info("Create engine successfully!")

     logger.info("Saving TRT engine file to path {}".format(save_path))
```

Model inferenece: `zetton::inference::vision::FastReIDTensorRTInferenceModel`

Example: `zetton-inference-tensorrt/examples/run_fast_reid.cc`

## Acknowledgements

- [Offical PyTorch Implementation of FastReID](https://github.com/JDAI-CV/fast-reid)
