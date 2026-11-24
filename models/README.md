# Model artifacts

The native runtime expects:

    models/onnx/rtmdet_nano.onnx
    models/onnx/rtmpose_body.onnx
    models/onnx/rtmpose_hand.onnx
    models/tensorrt/rtmdet_nano.engine
    models/tensorrt/rtmpose_body.engine
    models/tensorrt/rtmpose_hand.engine

Create ONNX with deploy/export_onnx_models.py, then build the FP16 engines on
the target Jetson with deploy/build_tensorrt_engines.sh. ONNX, checkpoint and
engine binaries are ignored by Git. TensorRT engines are not portable across
arbitrary JetPack/TensorRT/GPU combinations.

For the validated Jetson deployment used by this project, copy the known-good
detector/body/hand plans from the Realman source checkout and install the
desktop launcher in one step:

    bash scripts/install_b601_desktop_app.sh

The installer verifies SHA-256 after copying and atomically replaces the three
files under `models/tensorrt/`.
