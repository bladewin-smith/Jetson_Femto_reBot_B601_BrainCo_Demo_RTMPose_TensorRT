#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models}"
LOCAL_ENGINE_DIR="${B601_LOCAL_ENGINE_DIR:-$REPO/models/tensorrt_local}"
CHANNELS="${1:-[1.0,1.0,0.0,0.0,1.0,1.0]}"
VERIFIED_DET_ENGINE="$SOURCE_MODEL_DIR/tensorrt_orin_nano_20261102_170600/rtmdet_nano.engine"
VERIFIED_BODY_ENGINE="$SOURCE_MODEL_DIR/tensorrt_pose_fp32/rtmpose_body.engine"
LOCAL_HAND_ENGINE="$LOCAL_ENGINE_DIR/rtmpose_hand.engine"

if [[ -z "${TABLE_Q:-}" ]]; then
  echo "TABLE_Q is required and must be this arm's freshly measured table pose." >&2
  echo "Example: export TABLE_Q='[q1,q2,q3,q4,q5,q6]'" >&2
  exit 2
fi

# The hardware logs prove that the existing detector/body plans recognize the
# operator on this board without a compatibility warning.  Only the old hand
# plan produced TensorRT's cross-device warning, so replace that component with
# the target-local hand plan.  The newly rebuilt detector is kept opt-in until
# its numerical output has been validated on a real camera frame.
CUSTOM_ENGINE_COUNT=0
if [[ -n "${DET_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if [[ -n "${BODY_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if [[ -n "${HAND_ENGINE:-}" ]]; then ((CUSTOM_ENGINE_COUNT += 1)); fi
if (( CUSTOM_ENGINE_COUNT != 0 && CUSTOM_ENGINE_COUNT != 3 )); then
  echo "Set DET_ENGINE, BODY_ENGINE and HAND_ENGINE together, or unset all three." >&2
  exit 2
fi
if (( CUSTOM_ENGINE_COUNT == 3 )); then
  ENGINE_PROFILE="custom"
elif [[ "${B601_USE_ALL_LOCAL_ENGINES:-0}" == "1" \
     && -s "$LOCAL_ENGINE_DIR/rtmdet_nano.engine" \
     && -s "$LOCAL_ENGINE_DIR/rtmpose_body.engine" \
     && -s "$LOCAL_ENGINE_DIR/rtmpose_hand.engine" ]]; then
  DET_ENGINE="$LOCAL_ENGINE_DIR/rtmdet_nano.engine"
  BODY_ENGINE="$LOCAL_ENGINE_DIR/rtmpose_body.engine"
  HAND_ENGINE="$LOCAL_ENGINE_DIR/rtmpose_hand.engine"
  ENGINE_PROFILE="all-local-experimental"
elif [[ -s "$VERIFIED_DET_ENGINE" \
     && -s "$VERIFIED_BODY_ENGINE" \
     && -s "$LOCAL_HAND_ENGINE" ]]; then
  DET_ENGINE="$VERIFIED_DET_ENGINE"
  BODY_ENGINE="$VERIFIED_BODY_ENGINE"
  HAND_ENGINE="$LOCAL_HAND_ENGINE"
  ENGINE_PROFILE="verified-hybrid"
else
  echo "No validated B601 TensorRT engine profile is available." >&2
  echo "Expected the verified detector/body plans plus a target-local hand plan." >&2
  echo "Build the hand plan with: bash scripts/build_b601_local_tensorrt.sh" >&2
  exit 2
fi
for file in "$DET_ENGINE" "$BODY_ENGINE" "$HAND_ENGINE"; do
  if [[ ! -s "$file" ]]; then
    echo "Missing or empty TensorRT engine; refusing hardware startup: $file" >&2
    exit 2
  fi
done
if [[ "$ENGINE_PROFILE" == "all-local-experimental" ]]; then
  echo "WARNING: the all-local detector is experimental and failed the latest" >&2
  echo "         live person-detection test." >&2
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
echo "MIT stream: coordinated velocity interpolation + bounded position correction + adaptive frame hold"
echo "Workspace: Seeed published range intersected with current DM URDF, plus 0.10 rad margin"
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
  -p b601_safe_joint_lower_rad:='[-2.6005,-3.5954,-3.5954,-1.4992,-1.4992,-1.4992]' \
  -p b601_safe_joint_upper_rad:='[2.6005,0.0,0.0,1.4992,1.4992,1.4992]' \
  -p b601_joint_limit_margin_rad:=0.10 \
  -p b601_max_joint_target_delta_rad:='[0.045,0.065,0.125,0.075,0.012,0.105]' \
  -p b601_control_mode:=semantic_joint \
  -p b601_semantic_channel_enable:="$CHANNELS" \
  -p b601_semantic_joint_gains:='[3.00,-0.60,0.18,0.15,0.75]' \
  -p b601_semantic_max_joint_offset_rad:='[1.60,1.40,1.50,1.00,0.008,1.20]' \
  -p b601_semantic_max_step_rad:='[0.0036,0.0044,0.0056,0.0024,0.0008,0.0072]' \
  -p b601_semantic_max_tracking_error_rad:='[0.032,0.055,0.110,0.060,0.008,0.090]' \
  -p b601_semantic_vertical_max_lift_m:=0.180 \
  -p b601_semantic_planar_max_forward_m:=0.150 \
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
  -p b601_semantic_planar_command_max_acceleration_m_s2:='[0.24,0.12]' \
  -p b601_semantic_planar_command_deadband_m:='[0.0012,0.0015]' \
  -p b601_semantic_planar_idle_target_deadband_m:='[0.0025,0.0030]' \
  -p b601_semantic_planar_idle_velocity_deadband_m_s:='[0.0030,0.0030]' \
  -p b601_semantic_planar_idle_confirm_samples:=3 \
  -p b601_semantic_planar_tracking_error_m:='[0.030,0.050]' \
  -p b601_semantic_planar_backpressure_start_fraction:=0.65 \
  -p b601_semantic_planar_backpressure_rates_s:='[1.00,1.50]' \
  -p b601_semantic_clearance_gate_rates_s:='[1.5,3.0]' \
  -p b601_semantic_clearance_gate_deadband:=0.03 \
  -p b601_semantic_planar_follower_min_mobility_scale:=0.30 \
  -p b601_semantic_planar_task_error_gain_s:='[2.0,2.0]' \
  -p b601_semantic_planar_task_correction_max_velocity_m_s:='[0.025,0.020]' \
  -p b601_semantic_planar_joint_max_acceleration_rad_s2:='[0.36,0.48,0.15]' \
  -p b601_semantic_j1_target_max_velocity_rad_s:=0.10 \
  -p b601_semantic_j1_target_max_acceleration_rad_s2:=0.35 \
  -p b601_semantic_j1_target_deadband_rad:=0.0015 \
  -p b601_semantic_j4_target_max_velocity_rad_s:=0.045 \
  -p b601_semantic_j4_target_max_acceleration_rad_s2:=0.20 \
  -p b601_semantic_j4_target_deadband_rad:=0.0020 \
  -p b601_semantic_vertical_descent_guard_enabled:=true \
  -p b601_semantic_vertical_descent_intent_deadband_m_s:=0.0015 \
  -p b601_semantic_vertical_descent_guard_clearance_margin_m:=0.010 \
  -p b601_wrist_twist_filter_alpha:=0.22 \
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
  -p b601_wrist_twist_activate_deg:=10.0 \
  -p b601_wrist_twist_release_deg:=6.0 \
  -p b601_wrist_twist_confirm_samples:=2 \
  -p b601_wrist_twist_reverse_confirm_samples:=5 \
  -p b601_wrist_twist_min_joint_offset_rad:=0.080 \
  -p b601_wrist_twist_soft_start_s:=0.20 \
  -p b601_semantic_j6_target_max_velocity_rad_s:=0.16 \
  -p b601_semantic_j6_target_max_acceleration_rad_s2:=0.75 \
  -p b601_semantic_linear_deadband_m:='[0.012,0.008,0.008]' \
  -p b601_stream_max_velocity_rad_s:='[0.25,0.30,0.35,0.15,0.10,0.30]' \
  -p b601_stream_max_acceleration_rad_s2:='[1.6,2.0,2.4,0.85,0.8,3.0]' \
  -p b601_stream_max_jerk_rad_s3:='[50.0,65.0,80.0,32.0,30.0,100.0]' \
  -p b601_stream_outer_velocity_scale:=1.30 \
  -p b601_stream_planar_outer_velocity_scale:=1.15 \
  -p b601_stream_goal_velocity_filter_alpha:=0.45 \
  -p b601_stream_goal_velocity_stale_cycles:=2.0 \
  -p b601_stream_goal_period_filter_alpha:=0.20 \
  -p b601_stream_goal_velocity_max_hold_s:=0.55 \
  -p b601_stream_goal_observed_period_max_s:=0.50 \
  -p b601_semantic_outer_dt_max_s:=0.35 \
  -p b601_stream_trajectory_dt_max_s:=0.012 \
  -p b601_stream_position_correction_gain_s:=4.0 \
  -p b601_stream_position_correction_max_fraction:=0.35 \
  -p b601_stream_planar_position_correction_min_fraction:=0.18 \
  -p b601_stream_planar_position_correction_max_fraction:=0.32 \
  -p control_hand_side:=right \
  -p control_hand_fallback_enabled:=false \
  -p hand_control_enabled:=false \
  -p auto_home_on_start:=false \
  -p auto_start_on_ready:=false
