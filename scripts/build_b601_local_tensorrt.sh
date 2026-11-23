#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCAL_ONNX_DIR="$REPO/models/onnx"
LEGACY_ONNX_DIR="$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models/onnx"
ENGINE_DIR="${B601_LOCAL_ENGINE_DIR:-$REPO/models/tensorrt_local}"

has_models() {
  local directory="$1"
  [[ -s "$directory/rtmdet_nano.onnx" \
     && -s "$directory/rtmpose_body.onnx" \
     && -s "$directory/rtmpose_hand.onnx" ]]
}

if [[ -n "${SOURCE_ONNX_DIR:-}" ]]; then
  ONNX_DIR="$SOURCE_ONNX_DIR"
elif has_models "$LOCAL_ONNX_DIR"; then
  ONNX_DIR="$LOCAL_ONNX_DIR"
elif has_models "$LEGACY_ONNX_DIR"; then
  ONNX_DIR="$LEGACY_ONNX_DIR"
else
  echo "The three ONNX models were not found." >&2
  echo "Set SOURCE_ONNX_DIR to the directory containing rtmdet_nano.onnx," >&2
  echo "rtmpose_body.onnx and rtmpose_hand.onnx." >&2
  exit 2
fi

if ! has_models "$ONNX_DIR"; then
  echo "SOURCE_ONNX_DIR is incomplete: $ONNX_DIR" >&2
  exit 2
fi

echo "Building B601 vision engines on this Jetson"
echo "ONNX source: $ONNX_DIR"
echo "Engine output: $ENGINE_DIR"
bash "$REPO/deploy/build_tensorrt_engines.sh" "$ONNX_DIR" "$ENGINE_DIR"

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
  echo "onnx_source=$ONNX_DIR"
} > "$ENGINE_DIR/build_info.txt"

echo "Target-local TensorRT profile is ready."
echo "scripts/run_b601_smooth_follow.sh will select it automatically."
