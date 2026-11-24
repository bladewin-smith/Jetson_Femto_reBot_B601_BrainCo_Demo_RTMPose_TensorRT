# Model artifacts

The native runtime expects:

    models/onnx/rtmdet_nano.onnx
    models/onnx/rtmpose_body.onnx
    models/onnx/rtmpose_hand.onnx
    models/tensorrt/rtmdet_nano.engine
    models/tensorrt/rtmpose_body.engine
    models/tensorrt/rtmpose_hand.engine

Create ONNX with deploy/export_onnx_models.py, then build the FP16 engines on
the target Jetson with deploy/build_tensorrt_engines.sh. ONNX, checkpoints and
experimental engine builds remain ignored by Git. The three exact runtime
plans listed above are allow-listed and stored with Git LFS, together with
`verified_engines.sha256`. Fetch them after cloning with `git lfs pull`.

TensorRT engines are not portable across arbitrary JetPack/TensorRT/GPU
combinations. Replace the LFS bundle only after validating all three plans on
the target Jetson.

For the validated Jetson deployment used by this project, copy the known-good
detector/body/hand plans from the Realman source checkout and install the
desktop launcher in one step:

    bash scripts/install_b601_desktop_app.sh

The installer verifies SHA-256 after copying and atomically replaces the three
files under `models/tensorrt/`. If the source checkout is absent, it validates
and uses the bundle already downloaded through Git LFS.
