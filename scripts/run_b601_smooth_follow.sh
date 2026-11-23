#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models}"
LOCAL_ENGINE_DIR="${B601_LOCAL_ENGINE_DIR:-$REPO/models/tensorrt_local}"
CHANNELS="${1:-[1.0,1.0,0.0,0.0,1.0,1.0]}"

if [[ -z "${TABLE_Q:-}" ]]; then
  echo "TABLE_Q is required and must be this arm's freshly measured table pose." >&2
  echo "Example: export TABLE_Q='[q1,q2,q3,q4,q5,q6]'" >&2
  exit 2
fi

# Prefer engines built in this repository on the current Jetson.  TensorRT
# plans are not portable between GPU/JetPack/TensorRT combinations; the old
# mixed RealMan paths remain only as a migration fallback. Explicit paths
# always take precedence.
CUSTOM_ENGINE_COUNT=0
if [[ -n "${DET_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if [[ -n "${BODY_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if [[ -n "${HAND_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if (( CUSTOM_ENGINE_COUNT != 0 && CUSTOM_ENGINE_COUNT != 3 )); then
  echo "Set DET_ENGINE, BODY_ENGINE and HAND_ENGINE together, or unset all three." >&2
  exit 2
fi
if [[ -s "$LOCAL_ENGINE_DIR/rtmdet_nano.engine" \
   && -s "$LOCAL_ENGINE_DIR/rtmpose_body.engine" \
   && -s "$LOCAL_ENGINE_DIR/rtmpose_hand.engine" ]]; then
  DEFAULT_DET_ENGINE="$LOCAL_ENGINE_DIR/rtmdet_nano.engine"
  DEFAULT_BODY_ENGINE="$LOCAL_ENGINE_DIR/rtmpose_body.engine"
  DEFAULT_HAND_ENGINE="$LOCAL_ENGINE_DIR/rtmpose_hand.engine"
  ENGINE_PROFILE="target-local"
else
  DEFAULT_DET_ENGINE="$SOURCE_MODEL_DIR/tensorrt_orin_nano_20261102_170600/rtmdet_nano.engine"
  DEFAULT_BODY_ENGINE="$SOURCE_MODEL_DIR/tensorrt_pose_fp32/rtmpose_body.engine"
  DEFAULT_HAND_ENGINE="$SOURCE_MODEL_DIR/tensorrt/rtmpose_hand.engine"
  ENGINE_PROFILE="legacy-fallback"
fi
DET_ENGINE="${DET_ENGINE:-$DEFAULT_DET_ENGINE}"
BODY_ENGINE="${BODY_ENGINE:-$DEFAULT_BODY_ENGINE}"
HAND_ENGINE="${HAND_ENGINE:-$DEFAULT_HAND_ENGINE}"
if (( CUSTOM_ENGINE_COUNT == 3 )); then
  ENGINE_PROFILE="custom"
fi
if [[ "$ENGINE_PROFILE" == "legacy-fallback" \
   && "${B601_ALLOW_LEGACY_ENGINES:-0}" != "1" ]]; then
  echo "Refusing legacy mixed TensorRT plans because they can add latency and" >&2
  echo "produce the cross-device compatibility warning seen in the test logs." >&2
  echo "Build target-local plans with: bash scripts/build_b601_local_tensorrt.sh" >&2
  echo "Temporary diagnostic override: export B601_ALLOW_LEGACY_ENGINES=1" >&2
  exit 2
fi
for file in "$DET_ENGINE" "$BODY_ENGINE" "$HAND_ENGINE"; do
  if [[ ! -s "$file" ]]; then
    echo "Missing or empty TensorRT engine; refusing hardware startup: $file" >&2
    exit 2
  fi
done
if [[ "$ENGINE_PROFILE" == "legacy-fallback" ]]; then
  echo "WARNING: legacy TensorRT diagnostic override is active." >&2
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi
export PYTHONPATH="$HOME/ws/pyorbbecsdk-2-main:$HOME/ws/brainco-hand-sdk-main/python:$REPO:$REPO/scripts:$REPO/third_party/reBotArm_control_py:${PYTHONPATH:-}"

echo "B601 continuous semantic follower"
echo "Planar control: resolved-rate J2/J3/J4 + task anti-windup + continuous clearance gate"
echo "Vertical safety: anatomical-wrist anchor + wrist-rotation rebase + no-descent guard"
echo "MIT stream: velocity feed-forward + bounded position correction + adaptive frame hold"
echo "Channels: $CHANNELS"
echo "Engine profile: $ENGINE_PROFILE"
echo "Detector: $DET_ENGINE"
echo "Body pose: $BODY_ENGINE"
echo "Hand pose: $HAND_ENGINE"
echo "Keep the arm table-supported before startup and keep the E-stop within reach."

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
  -p b601_control_mode:=semantic_joint \
  -p b601_semantic_channel_enable:="$CHANNELS" \
  -p b601_semantic_joint_gains:='[2.00,-0.60,0.18,0.15,0.75]' \
  -p b601_semantic_max_joint_offset_rad:='[0.70,0.45,0.55,0.35,0.008,0.25]' \
  -p b601_semantic_max_step_rad:='[0.0036,0.0044,0.0056,0.0024,0.0008,0.0064]' \
  -p b601_semantic_max_tracking_error_rad:='[0.018,0.035,0.075,0.030,0.008,0.055]' \
  -p b601_semantic_vertical_max_lift_m:=0.060 \
  -p b601_semantic_planar_max_forward_m:=0.055 \
  -p b601_semantic_vertical_solver_step_rad:=0.020 \
  -p b601_semantic_vertical_solver_iterations:=20 \
  -p arm_command_interval_s:=0.04 \
  -p b601_joint_kp:='[120.0,120.0,180.0,45.0,35.0,30.0]' \
  -p b601_joint_kd:='[8.0,8.0,8.0,5.0,2.0,2.0]' \
  -p b601_joint_effort_limit_nm:='[8.0,10.0,16.0,6.0,4.0,4.0]' \
  -p b601_semantic_planar_joint_mobility:='[0.90,1.00,0.35]' \
  -p b601_semantic_stall_error_rad:='[0.010,0.015,0.030,0.025,0.010,0.018]' \
  -p b601_semantic_stall_min_progress_rad:=0.0010 \
  -p b601_semantic_stall_timeout_s:=6.0 \
  -p palm_filter_alpha:=0.38 \
  -p palm_filter_fast_alpha:=0.72 \
  -p palm_filter_speed_scale_m_s:=0.30 \
  -p palm_filter_max_innovation_m:='[0.030,0.030,0.030]' \
  -p b601_semantic_planar_input_scale:='[0.45,1.0]' \
  -p b601_semantic_motion_reverse_deadband_m:='[0.0045,0.0035,0.0040]' \
  -p b601_semantic_motion_reverse_confirm_samples:=3 \
  -p b601_semantic_planar_command_max_velocity_m_s:='[0.060,0.040]' \
  -p b601_semantic_planar_command_max_acceleration_m_s2:='[0.32,0.16]' \
  -p b601_semantic_planar_command_deadband_m:='[0.0012,0.0015]' \
  -p b601_semantic_planar_tracking_error_m:='[0.025,0.025]' \
  -p b601_semantic_planar_backpressure_start_fraction:=0.55 \
  -p b601_semantic_planar_backpressure_rates_s:='[2.0,4.0]' \
  -p b601_semantic_clearance_gate_rates_s:='[1.5,3.0]' \
  -p b601_semantic_clearance_gate_deadband:=0.03 \
  -p b601_semantic_planar_follower_min_mobility_scale:=0.30 \
  -p b601_semantic_planar_task_error_gain_s:='[2.0,2.0]' \
  -p b601_semantic_planar_task_correction_max_velocity_m_s:='[0.025,0.020]' \
  -p b601_semantic_planar_joint_max_acceleration_rad_s2:='[0.45,0.60,0.20]' \
  -p b601_semantic_j1_target_max_velocity_rad_s:=0.10 \
  -p b601_semantic_j1_target_max_acceleration_rad_s2:=0.45 \
  -p b601_semantic_j1_target_deadband_rad:=0.0015 \
  -p b601_semantic_j4_target_max_velocity_rad_s:=0.045 \
  -p b601_semantic_j4_target_max_acceleration_rad_s2:=0.20 \
  -p b601_semantic_j4_target_deadband_rad:=0.0020 \
  -p b601_semantic_vertical_descent_guard_enabled:=true \
  -p b601_semantic_vertical_descent_intent_deadband_m_s:=0.0015 \
  -p b601_semantic_vertical_descent_guard_clearance_margin_m:=0.010 \
  -p b601_wrist_twist_filter_alpha:=0.18 \
  -p b601_wrist_twist_max_innovation_deg:=8.0 \
  -p b601_wrist_twist_max_raw_rate_deg_s:=180.0 \
  -p b601_wrist_twist_max_raw_jump_deg:=25.0 \
  -p b601_wrist_twist_translation_gate_m_s:=0.050 \
  -p b601_wrist_twist_stationary_confirm_samples:=3 \
  -p b601_vertical_wrist_decoupling_enabled:=true \
  -p b601_vertical_wrist_hold_rate_deg_s:=3.0 \
  -p b601_vertical_wrist_hold_min_delta_deg:=6.0 \
  -p b601_vertical_wrist_release_rate_deg_s:=1.0 \
  -p b601_vertical_wrist_release_samples:=5 \
  -p b601_vertical_wrist_release_lift_rate_m_s:=0.012 \
  -p b601_wrist_twist_activate_deg:=12.0 \
  -p b601_wrist_twist_release_deg:=6.0 \
  -p b601_wrist_twist_confirm_samples:=3 \
  -p b601_wrist_twist_reverse_confirm_samples:=6 \
  -p b601_wrist_twist_min_joint_offset_rad:=0.080 \
  -p b601_wrist_twist_soft_start_s:=0.30 \
  -p b601_semantic_j6_target_max_velocity_rad_s:=0.10 \
  -p b601_semantic_j6_target_max_acceleration_rad_s2:=0.40 \
  -p b601_semantic_linear_deadband_m:='[0.012,0.008,0.008]' \
  -p b601_stream_max_velocity_rad_s:='[0.25,0.30,0.35,0.15,0.10,0.25]' \
  -p b601_stream_max_acceleration_rad_s2:='[2.0,2.5,3.0,1.2,0.8,2.0]' \
  -p b601_stream_max_jerk_rad_s3:='[80.0,100.0,120.0,50.0,30.0,80.0]' \
  -p b601_stream_outer_velocity_scale:=1.40 \
  -p b601_stream_goal_velocity_filter_alpha:=0.45 \
  -p b601_stream_goal_velocity_stale_cycles:=2.0 \
  -p b601_stream_goal_period_filter_alpha:=0.20 \
  -p b601_stream_goal_velocity_max_hold_s:=0.20 \
  -p b601_stream_position_correction_gain_s:=5.0 \
  -p b601_stream_position_correction_max_fraction:=0.38 \
  -p control_hand_side:=right \
  -p control_hand_fallback_enabled:=false \
  -p hand_control_enabled:=false \
  -p auto_home_on_start:=false \
  -p auto_start_on_ready:=false
