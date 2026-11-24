#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_REPO="${1:-${B601_MODEL_SOURCE_REPO:-$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT}}"
TARGET_DIR="${B601_VERIFIED_ENGINE_DIR:-$REPO/models/tensorrt}"

declare -A SOURCES=(
  [rtmdet_nano.engine]="$SOURCE_REPO/models/tensorrt_orin_nano_20261102_170600/rtmdet_nano.engine"
  [rtmpose_body.engine]="$SOURCE_REPO/models/tensorrt_pose_fp32/rtmpose_body.engine"
  [rtmpose_hand.engine]="$SOURCE_REPO/models/tensorrt/rtmpose_hand.engine"
)

for name in rtmdet_nano.engine rtmpose_body.engine rtmpose_hand.engine; do
  source_file="${SOURCES[$name]}"
  if [[ ! -s "$source_file" ]]; then
    echo "Missing or empty verified model: $source_file" >&2
    exit 2
  fi
done

mkdir -p "$TARGET_DIR"
STAGE_DIR="$(mktemp -d "$TARGET_DIR/.verified-install.XXXXXX")"
cleanup() { rm -rf -- "$STAGE_DIR"; }
trap cleanup EXIT

echo "Copying verified TensorRT engines into: $TARGET_DIR"
for name in rtmdet_nano.engine rtmpose_body.engine rtmpose_hand.engine; do
  source_file="${SOURCES[$name]}"
  echo "  - $name <- $source_file" >&2
  cp -- "$source_file" "$STAGE_DIR/$name"
  chmod 0644 "$STAGE_DIR/$name"
  source_hash="$(sha256sum "$source_file" | awk '{print $1}')"
  copied_hash="$(sha256sum "$STAGE_DIR/$name" | awk '{print $1}')"
  if [[ "$source_hash" != "$copied_hash" ]]; then
    echo "Checksum mismatch while copying $name" >&2
    exit 2
  fi
  printf '%s  %s\n' "$copied_hash" "$name"
done > "$STAGE_DIR/verified_engines.sha256"

# Each replacement is an atomic rename inside the destination filesystem.
for name in rtmdet_nano.engine rtmpose_body.engine rtmpose_hand.engine; do
  mv -f -- "$STAGE_DIR/$name" "$TARGET_DIR/$name"
done
mv -f -- "$STAGE_DIR/verified_engines.sha256" "$TARGET_DIR/verified_engines.sha256"

chmod +x "$REPO/scripts/launch_b601_desktop.sh"
desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
desktop_dir="${desktop_dir:-$HOME/Desktop}"
applications_dir="$HOME/.local/share/applications"
mkdir -p "$desktop_dir" "$applications_dir"

write_desktop_file() {
  local destination="$1"
  cat > "$destination" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=reBot B601 Vision Control
Name[zh_CN]=reBot B601 视觉机械臂与灵巧手
Comment=Safe table-pose scan and visual arm/hand control
Comment[zh_CN]=安全扫描桌面姿态并启动视觉机械臂与灵巧手
Exec=$REPO/scripts/launch_b601_desktop.sh
Path=$REPO
Icon=applications-engineering
Terminal=true
Categories=Utility;Robotics;
StartupNotify=true
EOF
  chmod +x "$destination"
  gio set "$destination" metadata::trusted true >/dev/null 2>&1 || true
}

desktop_file="$desktop_dir/reBot-B601-Vision-Control.desktop"
menu_file="$applications_dir/rebot-b601-vision-control.desktop"
write_desktop_file "$desktop_file"
write_desktop_file "$menu_file"
update-desktop-database "$applications_dir" >/dev/null 2>&1 || true

echo
echo "Installation complete."
echo "Verified engines:"
ls -lh "$TARGET_DIR/rtmdet_nano.engine" \
  "$TARGET_DIR/rtmpose_body.engine" \
  "$TARGET_DIR/rtmpose_hand.engine"
echo "Desktop shortcut: $desktop_file"
echo "Double-click it only after the unpowered arm is fully supported by the table."
