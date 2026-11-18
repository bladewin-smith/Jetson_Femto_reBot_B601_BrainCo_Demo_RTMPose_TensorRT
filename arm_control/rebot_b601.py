"""Seeed Studio reBot Arm B601-DM palm-following backend.

The original demo drove a RealMan RM65-B through its controller-level Cartesian
API.  The B601-DM has no equivalent controller and its joints do not have the
RM65-B brake arrangement.  This backend therefore owns the complete safety and
control path:

* require an operator-confirmed, table-supported startup pose;
* read and validate all six joints before enabling torque;
* hold the measured pose while MIT mode and gravity feed-forward start;
* solve every Cartesian target with the B601 Pinocchio model;
* stream bounded joint targets plus ``g(q)`` at the MotorBridge rate;
* keep negative-Z motion out of the table and require a lift before XY motion;
* return to the configured table pose with a minimum-jerk joint trajectory.

The Seeed SDK is imported lazily.  Vision-only mode therefore works without
MotorBridge, Pinocchio, or physical robot hardware installed.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class StartupGuardError(RuntimeError):
    """Raised before torque enable when the B601 startup checks fail."""


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} must contain {size} values, got {result.size}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = _vector(values, size, name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} values must all be greater than zero")
    return result


def validate_tabletop_startup(
    positions_rad: Sequence[float],
    velocities_rad_s: Sequence[float],
    tabletop_reference_rad: Sequence[float],
    tolerance_rad: Sequence[float] | float,
    max_abs_velocity_rad_s: float,
) -> None:
    """Fail closed unless the measured arm is stationary at its table pose."""

    positions = _vector(positions_rad, 6, "positions_rad")
    velocities = _vector(velocities_rad_s, 6, "velocities_rad_s")
    reference = _vector(tabletop_reference_rad, 6, "tabletop_reference_rad")
    if np.isscalar(tolerance_rad):
        tolerance = np.full(6, float(tolerance_rad), dtype=np.float64)
    else:
        tolerance = _positive_vector(tolerance_rad, 6, "tolerance_rad")
    if not np.isfinite(max_abs_velocity_rad_s) or max_abs_velocity_rad_s <= 0.0:
        raise ValueError("max_abs_velocity_rad_s must be positive and finite")

    pose_error = np.abs(positions - reference)
    moving = np.abs(velocities) > float(max_abs_velocity_rad_s)
    outside = pose_error > tolerance
    if np.any(moving) or np.any(outside):
        details = {
            "positions_rad": positions.round(5).tolist(),
            "velocities_rad_s": velocities.round(5).tolist(),
            "tabletop_reference_rad": reference.round(5).tolist(),
            "pose_error_rad": pose_error.round(5).tolist(),
            "tolerance_rad": tolerance.round(5).tolist(),
            "moving_joints": (np.flatnonzero(moving) + 1).tolist(),
            "outside_pose_joints": (np.flatnonzero(outside) + 1).tolist(),
        }
        raise StartupGuardError(
            "Refusing to enable B601-DM torque: the arm is not stationary in "
            f"the configured table-supported pose. {details}"
        )


def apply_tabletop_motion_envelope(
    offset_m: Sequence[float],
    negative_limits_m: Sequence[float],
    positive_limits_m: Sequence[float],
    min_lift_before_xy_m: float,
) -> np.ndarray:
    """Clamp an XYZ offset and inhibit lateral motion until the arm is lifted."""

    offset = _vector(offset_m, 3, "offset_m")
    negative = np.maximum(_vector(negative_limits_m, 3, "negative_limits_m"), 0.0)
    positive = np.maximum(_vector(positive_limits_m, 3, "positive_limits_m"), 0.0)
    bounded = np.minimum(np.maximum(offset, -negative), positive)
    if bounded[2] < max(0.0, float(min_lift_before_xy_m)):
        bounded[:2] = 0.0
    return bounded


def minimum_jerk_blend(progress: float) -> float:
    """Return the quintic minimum-jerk blend for progress in ``[0, 1]``."""

    s = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


class B601PalmFollower:
    """B601-DM implementation of the palm-follower interface used by the UI."""

    ARM_DOF = 6

    def __init__(self, node: Any):
        self.node = node
        self.logger = node.get_logger()
        self.enabled = bool(node.declare_parameter("robot_motion_enabled", False).value)

        repository_root = Path(__file__).resolve().parents[1]
        default_sdk_root = repository_root / "third_party" / "reBotArm_control_py"
        default_hardware_yaml = repository_root / "config" / "rebotarm_dm_brainco.yaml"
        default_urdf = (
            default_sdk_root / "urdf" / "DM" / "urdf" / "ReBot_Arm_DM.urdf"
        )
        self.sdk_root = Path(
            str(node.declare_parameter("b601_sdk_path", str(default_sdk_root)).value)
        ).expanduser()
        self.hardware_yaml = Path(
            str(
                node.declare_parameter(
                    "b601_hardware_yaml", str(default_hardware_yaml)
                ).value
            )
        ).expanduser()
        self.urdf_path = Path(
            str(node.declare_parameter("b601_urdf_path", str(default_urdf)).value)
        ).expanduser()
        self.end_effector_frame = str(
            node.declare_parameter("b601_end_effector_frame", "end_link").value
        ).strip()

        self.tabletop_confirmed = bool(
            node.declare_parameter("b601_tabletop_startup_confirmed", False).value
        )
        self.tabletop_joints = _vector(
            node.declare_parameter(
                "b601_tabletop_joints_rad", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).value,
            self.ARM_DOF,
            "b601_tabletop_joints_rad",
        )
        self.tabletop_tolerance = _positive_vector(
            node.declare_parameter(
                "b601_tabletop_joint_tolerance_rad",
                [0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
            ).value,
            self.ARM_DOF,
            "b601_tabletop_joint_tolerance_rad",
        )
        self.startup_max_velocity = float(
            node.declare_parameter("b601_startup_max_abs_velocity_rad_s", 0.20).value
        )
        self.feedback_timeout_s = float(
            node.declare_parameter("b601_feedback_timeout_s", 3.0).value
        )

        self.joint_kp = _positive_vector(
            node.declare_parameter(
                "b601_joint_kp", [120.0, 120.0, 120.0, 18.0, 18.0, 18.0]
            ).value,
            self.ARM_DOF,
            "b601_joint_kp",
        )
        self.joint_kd = _positive_vector(
            node.declare_parameter(
                "b601_joint_kd", [8.0, 8.0, 8.0, 2.0, 2.0, 2.0]
            ).value,
            self.ARM_DOF,
            "b601_joint_kd",
        )
        self.gravity_scale = _vector(
            node.declare_parameter(
                "b601_gravity_torque_scale", [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
            ).value,
            self.ARM_DOF,
            "b601_gravity_torque_scale",
        )
        self.joint_direction = _vector(
            node.declare_parameter(
                "b601_joint_direction", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ).value,
            self.ARM_DOF,
            "b601_joint_direction",
        )
        if not np.all(np.isin(self.joint_direction, (-1.0, 1.0))):
            raise ValueError("b601_joint_direction values must be either -1 or +1")
        self.torque_limit = _positive_vector(
            node.declare_parameter(
                "b601_gravity_torque_limit_nm", [8.0, 8.0, 8.0, 4.0, 4.0, 4.0]
            ).value,
            self.ARM_DOF,
            "b601_gravity_torque_limit_nm",
        )
        self.max_joint_target_delta = _positive_vector(
            node.declare_parameter(
                "b601_max_joint_target_delta_rad",
                [0.10, 0.10, 0.10, 0.14, 0.14, 0.14],
            ).value,
            self.ARM_DOF,
            "b601_max_joint_target_delta_rad",
        )
        self.joint_limit_margin_rad = float(
            node.declare_parameter("b601_joint_limit_margin_rad", 0.05).value
        )
        self.control_rate_hz = float(
            node.declare_parameter("b601_control_rate_hz", 500.0).value
        )

        self.auto_home_on_start = bool(
            node.declare_parameter("auto_home_on_start", False).value
        )
        if self.auto_home_on_start:
            raise ValueError(
                "auto_home_on_start must remain false for B601-DM. Place the disabled "
                "arm in its table-supported pose before startup; never command an "
                "unbraked arm from an unknown pose."
            )
        self.home_key_enabled = bool(node.declare_parameter("home_key_enabled", True).value)
        self.home_max_velocity_rad_s = float(
            node.declare_parameter("b601_home_max_velocity_rad_s", 0.20).value
        )
        self.home_send_hz = float(
            node.declare_parameter("b601_home_send_hz", 50.0).value
        )
        self.home_verify_tolerance_rad = float(
            node.declare_parameter("b601_home_verify_tolerance_rad", 0.06).value
        )
        self.home_timeout_s = float(
            node.declare_parameter("b601_home_timeout_s", 30.0).value
        )
        self.return_tabletop_on_exit = bool(
            node.declare_parameter("b601_return_tabletop_on_exit", True).value
        )

        self.arm_motion_unit_m = float(
            node.declare_parameter("arm_motion_unit_m", 0.005).value
        )
        self.palm_to_arm_scale = float(
            node.declare_parameter("palm_to_arm_scale", 1.0).value
        )
        self.arm_axis_scale = _vector(
            node.declare_parameter("arm_axis_scale", [1.0, 1.0, 1.0]).value,
            3,
            "arm_axis_scale",
        )
        self.arm_axis_deadband_m = _vector(
            node.declare_parameter("arm_axis_deadband_m", [0.003, 0.003, 0.003]).value,
            3,
            "arm_axis_deadband_m",
        )
        self.hand_deadband_m = float(
            node.declare_parameter("hand_deadband_m", 0.005).value
        )
        self.max_arm_axis_offset_m = _positive_vector(
            node.declare_parameter(
                "max_arm_axis_offset_m", [0.025, 0.025, 0.060]
            ).value,
            3,
            "max_arm_axis_offset_m",
        )
        self.workspace_negative_axis_offset_m = np.maximum(
            _vector(
                node.declare_parameter(
                    "b601_workspace_negative_axis_offset_m", [0.025, 0.025, 0.0]
                ).value,
                3,
                "b601_workspace_negative_axis_offset_m",
            ),
            0.0,
        )
        self.workspace_positive_axis_offset_m = np.minimum(
            self.max_arm_axis_offset_m,
            np.maximum(
                _vector(
                    node.declare_parameter(
                        "b601_workspace_positive_axis_offset_m", [0.025, 0.025, 0.060]
                    ).value,
                    3,
                    "b601_workspace_positive_axis_offset_m",
                ),
                0.0,
            ),
        )
        self.min_lift_before_xy_m = float(
            node.declare_parameter("b601_min_lift_before_xy_m", 0.020).value
        )
        self.max_arm_target_delta_axis_m = _positive_vector(
            node.declare_parameter(
                "max_arm_target_delta_axis_m", [0.003, 0.003, 0.003]
            ).value,
            3,
            "max_arm_target_delta_axis_m",
        )
        self.command_interval_s = float(
            node.declare_parameter("arm_command_interval_s", 0.06).value
        )
        self.min_command_delta_m = float(
            node.declare_parameter("min_command_delta_m", 0.0015).value
        )
        self.mit_control_enabled = bool(
            node.declare_parameter("arm_mit_control_enabled", True).value
        )
        self.mit_kp = float(node.declare_parameter("arm_mit_kp", 70.0).value)
        self.mit_kd = float(node.declare_parameter("arm_mit_kd", 16.0).value)
        self.arm_axis_max_velocity_m_s = _positive_vector(
            node.declare_parameter(
                "arm_axis_max_velocity_m_s", [0.025, 0.025, 0.025]
            ).value,
            3,
            "arm_axis_max_velocity_m_s",
        )
        self.arm_axis_max_accel_m_s2 = _positive_vector(
            node.declare_parameter(
                "arm_axis_max_accel_m_s2", [0.10, 0.10, 0.10]
            ).value,
            3,
            "arm_axis_max_accel_m_s2",
        )
        self.palm_filter_enabled = bool(
            node.declare_parameter("palm_filter_enabled", True).value
        )
        self.palm_filter_alpha = float(
            node.declare_parameter("palm_filter_alpha", 0.55).value
        )
        self.palm_filter_hold_deadband_m = float(
            node.declare_parameter("palm_filter_hold_deadband_m", 0.008).value
        )
        self.palm_filter_release_deadband_m = float(
            node.declare_parameter("palm_filter_release_deadband_m", 0.014).value
        )
        self.palm_filter_velocity_deadband_m_s = float(
            node.declare_parameter("palm_filter_velocity_deadband_m_s", 0.025).value
        )
        self.camera_to_arm_matrix = _vector(
            node.declare_parameter(
                "camera_to_arm_matrix",
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            ).value,
            9,
            "camera_to_arm_matrix",
        ).reshape(3, 3)

        self.robot = None
        self.sdk_lock = threading.RLock()
        self.target_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.home_done_event = threading.Event()
        self.worker = None
        self.initial_pose = None
        self.initial_rotation = None
        self.target_q = None
        self.last_measured_q = None
        self.last_measured_dq = None
        self.last_commanded_pose = None
        self.last_target_pose = None
        self.last_arm_offset = np.zeros(3, dtype=np.float64)
        self.last_submit_time = 0.0
        self.first_tracking_motion_ok = False
        self.tracking_cycle_started_at = 0.0
        self.home_requested = False
        self.home_motion_active = False
        self.last_home_succeeded = False
        self.motion_inflight = False
        self.motion_stage = None
        self.motion_started_at = 0.0
        self.last_control_error = None
        self.last_control_error_log_s = 0.0
        self.sdk_busy_log_interval_s = 2.0
        self.mit_position_offset = np.zeros(3, dtype=np.float64)
        self.mit_velocity_offset = np.zeros(3, dtype=np.float64)
        self.last_mit_update_time = 0.0
        self.palm_filtered_offset = np.zeros(3, dtype=np.float64)
        self.last_palm_raw_offset = None
        self.last_palm_filter_time = 0.0
        self.palm_motion_locked = True

        # Compatibility attributes read by the inherited direct-hand code.
        self.robot_ip = ""
        self.robot_port = 0
        self.robot_level = 0
        self.thread_mode = 0

        if not self.enabled:
            self.logger.warn(
                "B601-DM motion is disabled. Set robot_motion_enabled:=true and "
                "b601_tabletop_startup_confirmed:=true only after physically "
                "supporting the arm on the table."
            )
            return
        if not self.tabletop_confirmed:
            raise StartupGuardError(
                "B601-DM startup confirmation is missing. With power off, place the "
                "arm parallel to and fully supported by the table, clear the workspace, "
                "then launch with b601_tabletop_startup_confirmed:=true."
            )

        self._connect()
        self.worker = threading.Thread(
            target=self._motion_worker,
            name="b601-home-worker",
            daemon=True,
        )
        self.worker.start()

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        if not self.sdk_root.is_dir():
            raise FileNotFoundError(f"B601 SDK directory not found: {self.sdk_root}")
        if not self.hardware_yaml.is_file():
            raise FileNotFoundError(f"B601 hardware YAML not found: {self.hardware_yaml}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"B601 DM URDF not found: {self.urdf_path}")

        sdk_text = str(self.sdk_root.resolve())
        if sdk_text not in sys.path:
            sys.path.insert(0, sdk_text)

        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model
        from reBotArm_control_py.kinematics.inverse_kinematics import (
            IKParams,
            pos_rot_to_se3,
            solve_ik,
        )

        self._compute_fk = compute_fk
        self._compute_generalized_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._ik_params = IKParams(
            max_iter=250,
            tolerance=1e-4,
            step_size=0.45,
            damping=1e-5,
        )
        self._model = load_robot_model(str(self.urdf_path.resolve()))
        self._model_data = self._model.createData()
        frame_id = self._model.getFrameId(self.end_effector_frame)
        if frame_id >= self._model.nframes:
            raise ValueError(
                f"B601 end-effector frame {self.end_effector_frame!r} is absent from the DM URDF"
            )
        self._end_frame_id = frame_id
        self._joint_lower = self._model.lowerPositionLimit[: self.ARM_DOF].copy()
        self._joint_upper = self._model.upperPositionLimit[: self.ARM_DOF].copy()

        self.robot = RebotArm(str(self.hardware_yaml.resolve()))
        try:
            self.robot.connect()
            q, dq = self._read_complete_feedback()
            validate_tabletop_startup(
                q,
                dq,
                self.tabletop_joints,
                self.tabletop_tolerance,
                self.startup_max_velocity,
            )
            self._set_reference_from_q(q)
            self.target_q = q.copy()
            self.last_measured_q = q.copy()
            self.last_measured_dq = dq.copy()

            if not self.robot.arm.mode_mit(kp=self.joint_kp, kd=self.joint_kd):
                raise RuntimeError("One or more B601 joints failed to enter MIT mode")
            self.robot.disable_all()
            time.sleep(0.10)
            self.robot.enable_all()
            self.robot.start_control_loop(self._control_loop, rate=self.control_rate_hz)
            time.sleep(0.10)
            if not self.robot.control_loop_active or self.last_control_error is not None:
                raise RuntimeError(
                    "B601 MIT command stream did not start cleanly: "
                    f"{self.last_control_error}"
                )
        except Exception:
            if self.robot is not None:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                self.robot = None
            raise

        self.logger.info(
            "B601-DM enabled in table-supported pose: "
            f"q={q.round(4).tolist()} rad, TCP={self.initial_pose[:3]}, "
            f"control={self.control_rate_hz:.0f}Hz MIT + Pinocchio g(q)"
        )

    def _read_complete_feedback(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + max(0.1, self.feedback_timeout_s)
        while time.monotonic() < deadline:
            q, dq, _ = self.robot.get_state(request_feedback=True)
            q = np.asarray(q[: self.ARM_DOF], dtype=np.float64)
            dq = np.asarray(dq[: self.ARM_DOF], dtype=np.float64)
            states = [
                self.robot._motor_map[name].get_state()
                for name in self.robot.arm.joint_names
            ]
            if len(q) == self.ARM_DOF and all(state is not None for state in states):
                if np.all(np.isfinite(q)) and np.all(np.isfinite(dq)):
                    return q, dq
            time.sleep(0.03)
        raise StartupGuardError(
            "Timed out waiting for valid feedback from all six B601-DM joints; "
            "torque was not enabled. Check /dev/ttyACM*, IDs 0x01..0x06, feedback "
            "IDs 0x11..0x16, 921600 baud, and the 24 V supply."
        )

    def _control_loop(self, robot: Any, dt: float) -> None:
        del dt
        try:
            with self.sdk_lock:
                q = np.asarray(robot.arm.get_positions(), dtype=np.float64)[: self.ARM_DOF]
                dq = np.asarray(
                    robot.arm.get_velocities(request_feedback=False), dtype=np.float64
                )[: self.ARM_DOF]
                if q.size != self.ARM_DOF or not np.all(np.isfinite(q)):
                    raise RuntimeError(f"invalid joint feedback: {q}")
                with self.target_lock:
                    target_q = q.copy() if self.target_q is None else self.target_q.copy()
                tau = self._gravity_torque(q)
                robot.arm.send_mit(
                    target_q,
                    vel=np.zeros(self.ARM_DOF),
                    kp=self.joint_kp,
                    kd=self.joint_kd,
                    tau=tau,
                )
            self.last_measured_q = q
            self.last_measured_dq = dq
            self.last_control_error = None
        except Exception as exc:
            self.last_control_error = exc
            now = time.monotonic()
            if now - self.last_control_error_log_s >= self.sdk_busy_log_interval_s:
                self.logger.error(
                    f"B601 control-loop error; the last target remains active: {exc}"
                )
                self.last_control_error_log_s = now

    def _gravity_torque(self, q: np.ndarray) -> np.ndarray:
        q_for_model = q * self.joint_direction
        q_model = self._pad_q_for_model(self._model, q_for_model, self.ARM_DOF)
        tau = self._compute_generalized_gravity(
            self._model, q_model, self._model_data
        )[: self.ARM_DOF]
        tau = tau * self.joint_direction * self.gravity_scale
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def close(self) -> None:
        if not self.enabled or self.robot is None:
            return
        if self.return_tabletop_on_exit:
            self.request_home("shutdown")
            if not self.home_done_event.wait(timeout=max(1.0, self.home_timeout_s + 2.0)):
                self.logger.fatal(
                    "B601 did not confirm the table pose before shutdown. Support the arm "
                    "by hand and use the hardware emergency stop/power switch."
                )
            elif not self.last_home_succeeded:
                self.logger.fatal(
                    "B601 table-home failed verification before shutdown. Keep the arm "
                    "physically supported while the drive is disconnected."
                )

        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
            self.worker = None
        try:
            # Stop/join the SDK loop before taking sdk_lock. Holding this lock
            # while stop_control_loop() joins would deadlock if the callback was
            # waiting to enter its serialized bus section.
            self.robot.stop_control_loop()
            with self.sdk_lock:
                self.robot.disconnect()
            if self.last_home_succeeded or not self.return_tabletop_on_exit:
                self.logger.info("B601-DM disabled and disconnected after safe shutdown")
            else:
                self.logger.error(
                    "B601-DM disconnected without a verified table-home; physical support "
                    "is required"
                )
        finally:
            self.robot = None

    # ------------------------------------------------------------------
    # Tracking interface
    # ------------------------------------------------------------------

    def begin_tracking_cycle(self, reason: str = "person_start") -> bool:
        if not self.enabled:
            return True
        if self.last_control_error is not None or not self.robot.control_loop_active:
            self.logger.error(
                f"Cannot start B601 tracking: command stream unhealthy ({self.last_control_error})"
            )
            return False
        with self.sdk_lock:
            q, dq, _ = self.robot.get_state(request_feedback=True)
        q = np.asarray(q[: self.ARM_DOF], dtype=np.float64)
        dq = np.asarray(dq[: self.ARM_DOF], dtype=np.float64)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            self.logger.error("Cannot start B601 tracking: invalid joint feedback")
            return False
        with self.target_lock:
            self.target_q = q.copy()
        self._set_reference_from_q(q)
        self.last_target_pose = list(self.initial_pose)
        self.last_commanded_pose = list(self.initial_pose)
        self.last_arm_offset = np.zeros(3, dtype=np.float64)
        self.last_submit_time = 0.0
        self.first_tracking_motion_ok = False
        self.tracking_cycle_started_at = time.monotonic()
        self._reset_filters()
        self.logger.info(f"B601 Cartesian control reference reset ({reason})")
        return True

    def pause_control(self, reason: str = "tracking_paused") -> None:
        if not self.enabled or self.robot is None:
            return
        q = self.last_measured_q
        if q is not None:
            with self.target_lock:
                self.target_q = q.copy()
        self.last_target_pose = None
        self.last_submit_time = 0.0
        self._reset_filters()
        self.logger.info(
            f"B601 tracking paused; holding measured pose with gravity feed-forward ({reason})"
        )

    def update_from_palm_displacement(self, displacement: Sequence[float] | None):
        if not self.enabled or self.initial_pose is None or displacement is None:
            return None
        if self.home_requested or self.home_motion_active:
            return self.last_arm_offset

        camera_offset = _vector(displacement, 3, "palm displacement")
        arm_offset = self.camera_to_arm_matrix.dot(camera_offset)
        arm_offset *= self.palm_to_arm_scale
        arm_offset *= self.arm_axis_scale
        arm_offset[np.abs(arm_offset) < self.arm_axis_deadband_m] = 0.0
        if np.linalg.norm(arm_offset) < self.hand_deadband_m:
            arm_offset[:] = 0.0
        arm_offset = apply_tabletop_motion_envelope(
            arm_offset,
            self.workspace_negative_axis_offset_m,
            self.workspace_positive_axis_offset_m,
            self.min_lift_before_xy_m,
        )
        arm_offset = self._anti_drift_filter_offset(arm_offset)
        arm_offset = self._mit_filter_offset(arm_offset)
        if self.arm_motion_unit_m > 0.0:
            arm_offset = (
                np.round(arm_offset / self.arm_motion_unit_m) * self.arm_motion_unit_m
            )
        arm_offset = apply_tabletop_motion_envelope(
            arm_offset,
            self.workspace_negative_axis_offset_m,
            self.workspace_positive_axis_offset_m,
            self.min_lift_before_xy_m,
        )

        target_xyz = np.asarray(self.initial_pose[:3], dtype=np.float64) + arm_offset
        reference_xyz = (
            np.asarray(self.last_target_pose[:3], dtype=np.float64)
            if self.last_target_pose is not None
            else np.asarray(self.initial_pose[:3], dtype=np.float64)
        )
        delta = np.clip(
            target_xyz - reference_xyz,
            -self.max_arm_target_delta_axis_m,
            self.max_arm_target_delta_axis_m,
        )
        target_xyz = reference_xyz + delta
        arm_offset = target_xyz - np.asarray(self.initial_pose[:3], dtype=np.float64)

        now = time.monotonic()
        if now - self.last_submit_time < self.command_interval_s:
            return self.last_arm_offset
        if self.last_target_pose is not None:
            if np.linalg.norm(target_xyz - np.asarray(self.last_target_pose[:3])) < self.min_command_delta_m:
                return self.last_arm_offset

        q_seed = (
            self.last_measured_q.copy()
            if self.last_measured_q is not None
            else self.target_q.copy()
        )
        q_model = self._pad_q_for_model(self._model, q_seed, self.ARM_DOF)
        target = self._pos_rot_to_se3(target_xyz, rot=self.initial_rotation)
        result = self._solve_ik(
            self._model,
            self._model_data,
            self._end_frame_id,
            target,
            q_model,
            self._ik_params,
            controlled_joints=self.ARM_DOF,
            position_only=False,
        )
        if not result.success:
            self.logger.warn(
                f"Refusing B601 palm target: IK did not converge (error={result.error:.3e})"
            )
            return self.last_arm_offset

        q_target = np.asarray(result.q[: self.ARM_DOF], dtype=np.float64)
        if not self._joint_target_is_safe(q_target, q_seed):
            return self.last_arm_offset
        with self.target_lock:
            self.target_q = q_target
        pose = [*target_xyz.tolist(), *self.initial_pose[3:6]]
        self.last_target_pose = pose
        self.last_commanded_pose = pose
        self.last_submit_time = now
        self.last_arm_offset = arm_offset
        self.first_tracking_motion_ok = True
        return arm_offset

    def _joint_target_is_safe(self, target: np.ndarray, current: np.ndarray) -> bool:
        lower = self._joint_lower + self.joint_limit_margin_rad
        upper = self._joint_upper - self.joint_limit_margin_rad
        outside = np.logical_or(target < lower, target > upper)
        jump = np.abs(target - current) > self.max_joint_target_delta
        if np.any(outside) or np.any(jump):
            self.logger.warn(
                "Refusing B601 joint target: "
                f"outside_limits={(np.flatnonzero(outside) + 1).tolist()}, "
                f"excessive_step={(np.flatnonzero(jump) + 1).tolist()}, "
                f"target={target.round(4).tolist()}, current={current.round(4).tolist()}"
            )
            return False
        return True

    def _set_reference_from_q(self, q: np.ndarray) -> None:
        q_model = self._pad_q_for_model(self._model, q, self.ARM_DOF)
        position, rotation, _ = self._compute_fk(
            self._model, q_model, frame_name=self.end_effector_frame
        )
        try:
            import pinocchio as pin

            rpy = pin.rpy.matrixToRpy(rotation)
        except Exception:
            rpy = np.zeros(3, dtype=np.float64)
        self.initial_rotation = rotation.copy()
        self.initial_pose = [*position.tolist(), *rpy.tolist()]

    # ------------------------------------------------------------------
    # Table-home state machine
    # ------------------------------------------------------------------

    def request_home(self, reason: str = "manual") -> None:
        if not self.enabled:
            self.logger.warn("Ignoring table-home request because B601 motion is disabled")
            return
        self.home_done_event.clear()
        self.last_home_succeeded = False
        self.home_requested = True
        self.home_motion_active = True
        self.motion_stage = f"table_home:{reason}"
        self.logger.info(
            f"Queued slow B601 return to the configured table-supported pose ({reason})"
        )

    def _motion_worker(self) -> None:
        while not self.stop_event.is_set():
            if not self.home_requested:
                time.sleep(0.02)
                continue
            self.home_requested = False
            try:
                self._move_tabletop_home()
                self.last_home_succeeded = True
            except Exception as exc:
                self.last_home_succeeded = False
                self.logger.error(f"B601 table-home failed: {exc}")
            finally:
                self.home_motion_active = False
                self.motion_inflight = False
                self.motion_stage = None
                self.home_done_event.set()

    def _move_tabletop_home(self) -> None:
        q_start = (
            self.last_measured_q.copy()
            if self.last_measured_q is not None
            else np.asarray(self.robot.arm.get_positions(), dtype=np.float64)[: self.ARM_DOF]
        )
        max_error = float(np.max(np.abs(self.tabletop_joints - q_start)))
        if max_error <= self.home_verify_tolerance_rad:
            with self.target_lock:
                self.target_q = self.tabletop_joints.copy()
            return
        duration = max(1.0, 2.0 * max_error / max(self.home_max_velocity_rad_s, 0.02))
        steps = max(2, int(duration * max(self.home_send_hz, 10.0)))
        started = time.monotonic()
        self.motion_inflight = True
        self.motion_started_at = started
        self.logger.warn(
            "Returning B601 to table pose with a slow minimum-jerk joint path. "
            "Keep the table clear and be ready to support the arm."
        )
        for index in range(steps):
            if self.stop_event.is_set():
                raise RuntimeError("shutdown interrupted table-home motion")
            if time.monotonic() - started > self.home_timeout_s:
                raise TimeoutError("table-home motion exceeded its timeout")
            blend = minimum_jerk_blend((index + 1) / steps)
            target = q_start + blend * (self.tabletop_joints - q_start)
            with self.target_lock:
                self.target_q = target
            time.sleep(1.0 / max(self.home_send_hz, 10.0))
        time.sleep(0.30)
        with self.sdk_lock:
            measured = np.asarray(
                self.robot.arm.get_positions(), dtype=np.float64
            )[: self.ARM_DOF]
        error = float(np.max(np.abs(measured - self.tabletop_joints)))
        if error > self.home_verify_tolerance_rad:
            raise RuntimeError(
                f"table-home verification failed: max joint error={error:.4f} rad"
            )
        with self.target_lock:
            self.target_q = self.tabletop_joints.copy()
        self._set_reference_from_q(measured)
        self.last_arm_offset = np.zeros(3, dtype=np.float64)
        self.last_target_pose = list(self.initial_pose)
        self.last_commanded_pose = list(self.initial_pose)
        self._reset_filters()
        self.logger.info("B601 reached and verified the configured table-supported pose")

    def has_pending_or_active_motion(self) -> bool:
        return bool(self.enabled and (self.home_requested or self.home_motion_active))

    def warn_if_motion_stalled(self) -> None:
        now = time.monotonic()
        if (
            self.last_control_error is not None
            and now - self.last_control_error_log_s >= self.sdk_busy_log_interval_s
        ):
            self.logger.error(f"B601 command stream is unhealthy: {self.last_control_error}")
            self.last_control_error_log_s = now

    # ------------------------------------------------------------------
    # GUI-compatible filtering/tuning
    # ------------------------------------------------------------------

    def mit_tuning_values(self) -> dict[str, Any]:
        return {
            "arm_mit_control_enabled": bool(self.mit_control_enabled),
            "arm_mit_kp": float(self.mit_kp),
            "arm_mit_kd": float(self.mit_kd),
            "arm_axis_max_velocity_m_s": self.arm_axis_max_velocity_m_s.tolist(),
            "arm_axis_max_accel_m_s2": self.arm_axis_max_accel_m_s2.tolist(),
            "max_arm_axis_offset_m": self.max_arm_axis_offset_m.tolist(),
            "max_arm_target_delta_axis_m": self.max_arm_target_delta_axis_m.tolist(),
            "arm_command_interval_s": float(self.command_interval_s),
            "hand_deadband_m": float(self.hand_deadband_m),
            "arm_axis_deadband_m": self.arm_axis_deadband_m.tolist(),
            "palm_filter_alpha": float(self.palm_filter_alpha),
            "palm_filter_hold_deadband_m": float(self.palm_filter_hold_deadband_m),
            "palm_filter_release_deadband_m": float(self.palm_filter_release_deadband_m),
        }

    def apply_mit_tuning(self, values: dict[str, Any], reason: str = "runtime_tuning") -> None:
        self.mit_control_enabled = bool(
            values.get("arm_mit_control_enabled", self.mit_control_enabled)
        )
        self.mit_kp = max(1.0, float(values.get("arm_mit_kp", self.mit_kp)))
        self.mit_kd = max(0.0, float(values.get("arm_mit_kd", self.mit_kd)))
        self.command_interval_s = max(
            0.02, float(values.get("arm_command_interval_s", self.command_interval_s))
        )
        if "arm_axis_max_velocity_m_s" in values:
            self.arm_axis_max_velocity_m_s = _positive_vector(
                values["arm_axis_max_velocity_m_s"],
                3,
                "arm_axis_max_velocity_m_s",
            )
        if "arm_axis_max_accel_m_s2" in values:
            self.arm_axis_max_accel_m_s2 = _positive_vector(
                values["arm_axis_max_accel_m_s2"],
                3,
                "arm_axis_max_accel_m_s2",
            )
        if "max_arm_axis_offset_m" in values:
            self.max_arm_axis_offset_m = _positive_vector(
                values["max_arm_axis_offset_m"], 3, "max_arm_axis_offset_m"
            )
            self.workspace_negative_axis_offset_m = np.minimum(
                self.workspace_negative_axis_offset_m, self.max_arm_axis_offset_m
            )
            self.workspace_positive_axis_offset_m = np.minimum(
                self.workspace_positive_axis_offset_m, self.max_arm_axis_offset_m
            )
        if "max_arm_target_delta_axis_m" in values:
            self.max_arm_target_delta_axis_m = _positive_vector(
                values["max_arm_target_delta_axis_m"],
                3,
                "max_arm_target_delta_axis_m",
            )
        self.hand_deadband_m = max(
            0.0, float(values.get("hand_deadband_m", self.hand_deadband_m))
        )
        if "arm_axis_deadband_m" in values:
            self.arm_axis_deadband_m = _vector(
                values["arm_axis_deadband_m"], 3, "arm_axis_deadband_m"
            )
        self.palm_filter_alpha = float(
            np.clip(values.get("palm_filter_alpha", self.palm_filter_alpha), 0.0, 1.0)
        )
        self.palm_filter_hold_deadband_m = max(
            0.0,
            float(
                values.get(
                    "palm_filter_hold_deadband_m", self.palm_filter_hold_deadband_m
                )
            ),
        )
        self.palm_filter_release_deadband_m = max(
            self.palm_filter_hold_deadband_m,
            float(
                values.get(
                    "palm_filter_release_deadband_m",
                    self.palm_filter_release_deadband_m,
                )
            ),
        )
        self._reset_filters()
        self.logger.info(f"Applied B601 outer-loop tuning ({reason})")

    def _mit_filter_offset(self, desired: np.ndarray) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float64)
        if not self.mit_control_enabled:
            self.mit_position_offset = desired.copy()
            self.mit_velocity_offset[:] = 0.0
            self.last_mit_update_time = time.monotonic()
            return desired
        now = time.monotonic()
        if self.last_mit_update_time <= 0.0:
            self.last_mit_update_time = now
            self.mit_position_offset = desired.copy()
            self.mit_velocity_offset[:] = 0.0
            return desired
        dt = max(0.001, min(now - self.last_mit_update_time, 0.08))
        self.last_mit_update_time = now
        error = desired - self.mit_position_offset
        accel = self.mit_kp * error - self.mit_kd * self.mit_velocity_offset
        accel = np.clip(accel, -self.arm_axis_max_accel_m_s2, self.arm_axis_max_accel_m_s2)
        self.mit_velocity_offset += accel * dt
        self.mit_velocity_offset = np.clip(
            self.mit_velocity_offset,
            -self.arm_axis_max_velocity_m_s,
            self.arm_axis_max_velocity_m_s,
        )
        self.mit_position_offset += self.mit_velocity_offset * dt
        return self.mit_position_offset.copy()

    def _anti_drift_filter_offset(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if not self.palm_filter_enabled:
            return raw
        now = time.monotonic()
        if self.last_palm_raw_offset is None:
            self.last_palm_raw_offset = raw.copy()
            self.last_palm_filter_time = now
            self.palm_motion_locked = np.linalg.norm(raw) < self.palm_filter_release_deadband_m
            self.palm_filtered_offset = np.zeros(3) if self.palm_motion_locked else raw.copy()
            return self.palm_filtered_offset.copy()
        dt = max(0.001, min(now - self.last_palm_filter_time, 0.2))
        speed = np.linalg.norm(raw - self.last_palm_raw_offset) / dt
        self.last_palm_raw_offset = raw.copy()
        self.last_palm_filter_time = now
        norm = np.linalg.norm(raw)
        if norm < self.palm_filter_hold_deadband_m and speed < self.palm_filter_velocity_deadband_m_s:
            self.palm_motion_locked = True
        if self.palm_motion_locked:
            if norm >= self.palm_filter_release_deadband_m:
                self.palm_motion_locked = False
            else:
                self.palm_filtered_offset[:] = 0.0
                return self.palm_filtered_offset.copy()
        alpha = float(np.clip(self.palm_filter_alpha, 0.0, 1.0))
        self.palm_filtered_offset = alpha * raw + (1.0 - alpha) * self.palm_filtered_offset
        return self.palm_filtered_offset.copy()

    def _reset_filters(self) -> None:
        self.mit_position_offset = np.zeros(3, dtype=np.float64)
        self.mit_velocity_offset = np.zeros(3, dtype=np.float64)
        self.last_mit_update_time = 0.0
        self.palm_filtered_offset = np.zeros(3, dtype=np.float64)
        self.last_palm_raw_offset = None
        self.last_palm_filter_time = 0.0
        self.palm_motion_locked = True
