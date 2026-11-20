#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENGINE_DIR="${ENGINE_DIR:-$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models/tensorrt}"
CHANNELS="${1:-[0.0,0.0,0.0,0.0,0.0,1.0]}"

if [[ -z "${TABLE_Q:-}" ]]; then
  echo "TABLE_Q is required and must contain this arm's freshly measured six-joint table pose." >&2
  echo "Example: export TABLE_Q='[q1,q2,q3,q4,q5,q6]'" >&2
  exit 2
fi

# Preserve individually selected, already validated engines. This matters on
# Jetson systems where RTMDet and the two RTMPose engines were built with
# different precision profiles or live in different directories.
DET_ENGINE="${DET_ENGINE:-$ENGINE_DIR/rtmdet_nano.engine}"
BODY_ENGINE="${BODY_ENGINE:-$ENGINE_DIR/rtmpose_body.engine}"
HAND_ENGINE="${HAND_ENGINE:-$ENGINE_DIR/rtmpose_hand.engine}"
for file in "$DET_ENGINE" "$BODY_ENGINE" "$HAND_ENGINE"; do
  if [[ ! -s "$file" ]]; then
    echo "Missing or empty TensorRT engine; refusing hardware startup: $file" >&2
    exit 2
  fi
done

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # ROS Humble's generated setup scripts read several optional variables
  # before checking whether they are defined, which is incompatible with
  # `set -u`. Keep strict mode for this launcher, but suspend nounset only
  # while the trusted ROS environment script is being sourced.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi
export PYTHONPATH="$HOME/ws/pyorbbecsdk-2-main:$HOME/ws/brainco-hand-sdk-main/python:$REPO:$REPO/scripts:$REPO/third_party/reBotArm_control_py:${PYTHONPATH:-}"

echo "Visible B601 planar profile: J2/J3/J4 joint forward/height solve, 25 mm lift"
echo "Forward mapping: 0.35x palm depth, bounded to +/-35 mm after tabletop clearance"
echo "J4 anti-jitter: task slew limit + 25 mrad/s velocity + 0.10 rad/s^2 acceleration"
echo "J6 wrist twist: 3-D palm normal around the bent elbow-to-wrist axis; 9.2 deg max"
echo "For J6 keep the shoulder, bent elbow, wrist and open palm visible to the depth camera"
echo "Channels: $CHANNELS"
echo "Detector engine: $DET_ENGINE"
echo "Body-pose engine: $BODY_ENGINE"
echo "Hand-pose engine: $HAND_ENGINE"
echo "Keep the arm table-supported and the E-stop within immediate reach."

exec python "$REPO/scripts/camera_arm_b601_control_pose_hands.py" --ros-args \
  --params-file "$REPO/config/b601_visible_motion.yaml" \
  -p vision_backend:=rtmpose \
  -p rtmpose_runtime:=tensorrt \
  -p rtmdet_engine_path:="$DET_ENGINE" \
  -p rtmpose_body_engine_path:="$BODY_ENGINE" \
  -p rtmpose_hand_engine_path:="$HAND_ENGINE" \
  -p robot_motion_enabled:=true \
  -p b601_tabletop_startup_confirmed:=true \
  -p b601_tabletop_joints_rad:="$TABLE_Q" \
  -p b601_tabletop_joint_tolerance_rad:='[0.12,0.12,0.12,0.12,0.12,0.12]' \
  -p b601_semantic_channel_enable:="$CHANNELS" \
  -p control_hand_side:=right \
  -p control_hand_fallback_enabled:=false \
  -p hand_control_enabled:=false \
  -p auto_home_on_start:=false \
  -p auto_start_on_ready:=false
