#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCAL_ONNX_DIR="$REPO/models/onnx"
LEGACY_ONNX_DIR="$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models/onnx"
ENGINE_DIR="${B601_LOCAL_ENGINE_DIR:-$REPO/models/tensorrt_local}"
ENGINE_SET="${B601_ENGINE_SET:-hand}"

case "$ENGINE_SET" in
  hand)
    REQUIRED_MODELS=(rtmpose_hand)
    BUILD_MODELS=(rtmpose_hand)
    BUILD_OPT_LEVEL="${TRT_BUILDER_OPT_LEVEL:-2}"
    AVG_TIMING="${TRT_AVG_TIMING:-1}"
    ;;
  all)
    REQUIRED_MODELS=(rtmdet_nano rtmpose_body rtmpose_hand)
    BUILD_MODELS=(rtmdet_nano rtmpose_body rtmpose_hand)
    BUILD_OPT_LEVEL="${TRT_BUILDER_OPT_LEVEL:-3}"
    AVG_TIMING="${TRT_AVG_TIMING:-8}"
    ;;
  *)
    echo "B601_ENGINE_SET must be 'hand' or 'all': $ENGINE_SET" >&2
    exit 2
    ;;
esac

has_models() {
  local directory="$1"
  local model_name
  for model_name in "${REQUIRED_MODELS[@]}"; do
    [[ -s "$directory/$model_name.onnx" ]] || return 1
  done
  return 0
}

if [[ -n "${SOURCE_ONNX_DIR:-}" ]]; then
  ONNX_DIR="$SOURCE_ONNX_DIR"
elif has_models "$LOCAL_ONNX_DIR"; then
  ONNX_DIR="$LOCAL_ONNX_DIR"
elif has_models "$LEGACY_ONNX_DIR"; then
  ONNX_DIR="$LEGACY_ONNX_DIR"
else
  echo "The required ONNX model set was not found: $ENGINE_SET" >&2
  echo "Set SOURCE_ONNX_DIR to the ONNX model directory." >&2
  exit 2
fi

if ! has_models "$ONNX_DIR"; then
  echo "SOURCE_ONNX_DIR is incomplete: $ONNX_DIR" >&2
  exit 2
fi

echo "Building B601 vision engine set on this Jetson: $ENGINE_SET"
echo "ONNX source: $ONNX_DIR"
echo "Engine output: $ENGINE_DIR"
TRT_BUILDER_OPT_LEVEL="$BUILD_OPT_LEVEL" \
TRT_AVG_TIMING="$AVG_TIMING" \
  bash "$REPO/deploy/build_tensorrt_engines.sh" \
    "$ONNX_DIR" "$ENGINE_DIR" "${BUILD_MODELS[@]}"

{
  echo "built_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srmo)"
  if [[ -r /etc/nv_tegra_release ]]; then
    echo "jetpack=$(head -n 1 /etc/nv_tegra_release)"
  fi
  if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W -f='tensorrt=${Version}\n' tensorrt 2>/dev/null || true
    dpkg-query -W -f='libnvinfer=${Version}\n' libnvinfer10 2>/dev/null || true
  fi
  echo "engine_set=$ENGINE_SET"
  echo "builder_optimization_level=$BUILD_OPT_LEVEL"
  echo "avg_timing=$AVG_TIMING"
  echo "onnx_source=$ONNX_DIR"
} > "$ENGINE_DIR/build_info.txt"

echo "Target-local TensorRT engine set is ready: $ENGINE_SET"
echo "scripts/run_b601_smooth_follow.sh will select the validated profile."
