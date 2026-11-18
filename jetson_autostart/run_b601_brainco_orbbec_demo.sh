#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${B601_DEMO_ENV_FILE:-${SCRIPT_DIR}/b601_brainco_orbbec_demo.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; copy b601_brainco_orbbec_demo.env.example first" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

APP_ROOT="${APP_ROOT:-/home/jetson/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT}"
CONTROL_SCRIPT="${CONTROL_SCRIPT:-${APP_ROOT}/scripts/camera_arm_b601_control_pose_hands.py}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

is_true() { [[ "${1:-false}" =~ ^(true|True|TRUE|1|yes|YES)$ ]]; }
fail() { echo "[b601-demo] ERROR: $*" >&2; exit 2; }
add_param() { ARGS+=("-p" "$1:=$2"); }

[[ -f "${ROS_SETUP}" ]] || fail "ROS setup not found: ${ROS_SETUP}"
[[ -f "${CONTROL_SCRIPT}" ]] || fail "control script not found: ${CONTROL_SCRIPT}"
for model in "${RTMDET_ENGINE_PATH:-}" "${RTMPOSE_BODY_ENGINE_PATH:-}" "${RTMPOSE_HAND_ENGINE_PATH:-}"; do
  [[ -f "${model}" ]] || fail "TensorRT engine not found: ${model:-<empty>}"
done

if is_true "${ROBOT_MOTION_ENABLED:-false}"; then
  is_true "${B601_TABLETOP_STARTUP_CONFIRMED:-false}" || fail \
    "motion requested without B601_TABLETOP_STARTUP_CONFIRMED=true"
  [[ "${AUTO_HOME_ON_START:-false}" == "false" ]] || fail \
    "AUTO_HOME_ON_START must remain false for the unbraked B601-DM"
  [[ -e "${B601_MOTORBRIDGE_DEVICE:-/dev/ttyACM0}" ]] || fail \
    "MotorBridge device not found: ${B601_MOTORBRIDGE_DEVICE:-/dev/ttyACM0}"
fi

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYORBBEC_ROOT:-}:${BRAINCO_HAND_SDK_PATH:-}:${APP_ROOT}:${APP_ROOT}/scripts:${PYTHONPATH:-}"
cd "${APP_ROOT}"

ARGS=("${CONTROL_SCRIPT}" "--ros-args")
add_param vision_backend "${VISION_BACKEND:-rtmpose}"
add_param rtmpose_runtime "${RTMPOSE_RUNTIME:-tensorrt}"
add_param rtmdet_engine_path "${RTMDET_ENGINE_PATH}"
add_param rtmpose_body_engine_path "${RTMPOSE_BODY_ENGINE_PATH}"
add_param rtmpose_hand_engine_path "${RTMPOSE_HAND_ENGINE_PATH}"
add_param rtmpose_body_keypoint_threshold "${RTMPOSE_BODY_KEYPOINT_THRESHOLD:-0.25}"
add_param rtmpose_hand_keypoint_threshold "${RTMPOSE_HAND_KEYPOINT_THRESHOLD:-0.25}"
add_param control_hand_side "${CONTROL_HAND_SIDE:-right}"

add_param robot_motion_enabled "${ROBOT_MOTION_ENABLED:-false}"
add_param b601_tabletop_startup_confirmed "${B601_TABLETOP_STARTUP_CONFIRMED:-false}"
add_param b601_tabletop_joints_rad "${B601_TABLETOP_JOINTS_RAD:-[0,0,0,0,0,0]}"
add_param b601_tabletop_joint_tolerance_rad "${B601_TABLETOP_JOINT_TOLERANCE_RAD:-[0.12,0.12,0.12,0.12,0.12,0.12]}"
add_param b601_hardware_yaml "${B601_HARDWARE_YAML:-${APP_ROOT}/config/rebotarm_dm_brainco.yaml}"
add_param b601_urdf_path "${B601_URDF_PATH:-${APP_ROOT}/third_party/reBotArm_control_py/urdf/DM/urdf/ReBot_Arm_DM.urdf}"
add_param auto_home_on_start false
add_param home_key_enabled true
add_param b601_return_tabletop_on_exit true
add_param b601_workspace_negative_axis_offset_m "${B601_WORKSPACE_NEGATIVE_AXIS_OFFSET_M:-[0.025,0.025,0.0]}"
add_param b601_workspace_positive_axis_offset_m "${B601_WORKSPACE_POSITIVE_AXIS_OFFSET_M:-[0.025,0.025,0.060]}"
add_param b601_min_lift_before_xy_m "${B601_MIN_LIFT_BEFORE_XY_M:-0.020}"
add_param max_arm_axis_offset_m "${MAX_ARM_AXIS_OFFSET_M:-[0.025,0.025,0.060]}"
add_param max_arm_target_delta_axis_m "${MAX_ARM_TARGET_DELTA_AXIS_M:-[0.003,0.003,0.003]}"
add_param arm_axis_max_velocity_m_s "${ARM_AXIS_MAX_VELOCITY_M_S:-[0.025,0.025,0.025]}"
add_param arm_axis_max_accel_m_s2 "${ARM_AXIS_MAX_ACCEL_M_S2:-[0.10,0.10,0.10]}"
add_param arm_motion_unit_m "${ARM_MOTION_UNIT_M:-0.005}"
add_param arm_command_interval_s "${ARM_COMMAND_INTERVAL_S:-0.06}"
add_param palm_to_arm_scale "${PALM_TO_ARM_SCALE:-0.20}"

add_param hand_control_enabled "${HAND_CONTROL_ENABLED:-false}"
add_param hand_transport "${HAND_TRANSPORT:-brainco-rs485}"
add_param hand_rs485_sdk_path "${BRAINCO_HAND_SDK_PATH:-}"
add_param hand_rs485_port "${HAND_RS485_PORT:-auto}"
add_param hand_rs485_slave_id "${HAND_RS485_SLAVE_ID:-0}"
add_param hand_rs485_baudrate "${HAND_RS485_BAUDRATE:-460800}"
add_param hand_configure_tool_voltage false

add_param auto_start_on_ready "${AUTO_START_ON_READY:-false}"
add_param show_window "${SHOW_WINDOW:-true}"

echo "[b601-demo] Starting in manual-S mode; H returns slowly to the table pose"
exec "${PYTHON_BIN}" "${ARGS[@]}"
