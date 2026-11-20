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

from .semantic_motion import (
    OperatorArmMotion,
    semantic_joint_offsets,
    wrap_angle_radians,
)


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


def joint_target_safety_violations(
    target_rad: Sequence[float],
    current_rad: Sequence[float],
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    margin_rad: float,
    max_delta_rad: Sequence[float],
    margin_reference_rad: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard/margin-limit and per-command step violation masks.

    The B601-DM table-supported pose intentionally places joints 2 and 3 near
    their URDF upper limit (zero radians).  A conventional symmetric margin
    would make that valid startup pose a trap: even a target moving farther
    into the legal range would be rejected.  For a joint whose validated
    startup reference is already inside a margin zone, that reference replaces
    the normal guard boundary.  Tracking may therefore move inward and return
    to the validated table pose, but may not move any closer to the hard stop.
    The URDF hard limits are never relaxed.
    """

    target = np.asarray(target_rad, dtype=np.float64).reshape(-1)
    current = np.asarray(current_rad, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower_rad, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_rad, dtype=np.float64).reshape(-1)
    max_delta = np.asarray(max_delta_rad, dtype=np.float64).reshape(-1)
    margin_reference = (
        current
        if margin_reference_rad is None
        else np.asarray(margin_reference_rad, dtype=np.float64).reshape(-1)
    )
    if not (
        target.shape
        == current.shape
        == lower.shape
        == upper.shape
        == max_delta.shape
        == margin_reference.shape
    ):
        raise ValueError("joint safety vectors must have matching shapes")
    margin = float(margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")

    hard_limit = np.logical_or(target < lower, target > upper)
    lower_guard, upper_guard = joint_tracking_bounds(
        lower,
        upper,
        margin,
        margin_reference,
    )
    numerical_tolerance = 1e-6
    outside_margin = np.logical_or(
        target < lower_guard - numerical_tolerance,
        target > upper_guard + numerical_tolerance,
    )
    outside = hard_limit | outside_margin
    jump = np.abs(target - current) > max_delta
    return outside, jump


def joint_tracking_bounds(
    lower_rad: Sequence[float],
    upper_rad: Sequence[float],
    margin_rad: float,
    margin_reference_rad: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return effective tracking bounds without trapping a valid table pose."""

    lower = np.asarray(lower_rad, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_rad, dtype=np.float64).reshape(-1)
    reference = np.asarray(margin_reference_rad, dtype=np.float64).reshape(-1)
    if not lower.shape == upper.shape == reference.shape:
        raise ValueError("joint tracking-bound vectors must have matching shapes")
    margin = float(margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")

    nominal_lower_guard = lower + margin
    nominal_upper_guard = upper - margin
    lower_guard = np.where(
        reference < nominal_lower_guard,
        reference,
        nominal_lower_guard,
    )
    upper_guard = np.where(
        reference > nominal_upper_guard,
        reference,
        nominal_upper_guard,
    )
    return np.maximum(lower_guard, lower), np.minimum(upper_guard, upper)


def scale_vector_to_max_abs(
    values: Sequence[float], max_abs_value: float
) -> np.ndarray:
    """Limit a vector without changing its coordinated joint direction."""

    result = np.asarray(values, dtype=np.float64).reshape(-1)
    maximum = float(max_abs_value)
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinated joint step must contain only finite values")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_abs_value must be positive and finite")
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > maximum:
        result = result * (maximum / peak)
    return result


def ramp_joint_target_with_tracking_window(
    desired_rad: Sequence[float],
    measured_rad: Sequence[float],
    previous_target_rad: Sequence[float],
    max_step_rad: Sequence[float],
    max_tracking_error_rad: Sequence[float],
) -> np.ndarray:
    """Ramp from the prior target while bounding target lead over feedback."""

    desired = np.asarray(desired_rad, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_rad, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_target_rad, dtype=np.float64).reshape(-1)
    max_step = np.asarray(max_step_rad, dtype=np.float64).reshape(-1)
    max_tracking_error = np.asarray(
        max_tracking_error_rad, dtype=np.float64
    ).reshape(-1)
    if not (
        desired.shape
        == measured.shape
        == previous.shape
        == max_step.shape
        == max_tracking_error.shape
    ):
        raise ValueError("semantic joint-ramp vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(values))
        for values in (desired, measured, previous, max_step, max_tracking_error)
    ):
        raise ValueError("semantic joint-ramp vectors must contain finite values")
    if np.any(max_step <= 0.0) or np.any(max_tracking_error <= 0.0):
        raise ValueError("semantic joint-ramp limits must be positive")

    ramped = previous + np.clip(desired - previous, -max_step, max_step)
    return measured + np.clip(
        ramped - measured,
        -max_tracking_error,
        max_tracking_error,
    )


def resolve_native_frame_ids(model: Any, frame_names: Sequence[str]) -> list[int]:
    """Resolve Pinocchio frame IDs as native Python integers."""

    resolved = [int(model.getFrameId(str(name))) for name in frame_names]
    if any(frame_id < 0 or frame_id >= int(model.nframes) for frame_id in resolved):
        raise ValueError(
            "B601 DM URDF is missing one or more required semantic-control frames"
        )
    return resolved


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
        if (
            not np.isfinite(self.joint_limit_margin_rad)
            or self.joint_limit_margin_rad < 0.0
        ):
            raise ValueError("b601_joint_limit_margin_rad must be finite and non-negative")
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

        self.control_mode = str(
            node.declare_parameter("b601_control_mode", "cartesian").value
        ).strip().lower()
        if self.control_mode not in ("cartesian", "semantic_joint"):
            raise ValueError(
                "b601_control_mode must be 'cartesian' or 'semantic_joint'"
            )
        # Channels: J1 palm lateral, J2 palm depth, J3 elbow, J4 wrist,
        # J6 palm roll, and coordinated J2/J3/J4 vertical translation.
        self.semantic_channel_enable = _vector(
            node.declare_parameter(
                "b601_semantic_channel_enable", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ).value,
            6,
            "b601_semantic_channel_enable",
        )
        if not np.all(np.isin(self.semantic_channel_enable, (0.0, 1.0))):
            raise ValueError("b601_semantic_channel_enable values must be 0 or 1")
        self.semantic_joint_gains = _vector(
            node.declare_parameter(
                "b601_semantic_joint_gains",
                [0.60, -0.60, 0.20, 0.15, 0.20],
            ).value,
            5,
            "b601_semantic_joint_gains",
        )
        self.semantic_linear_deadband_m = np.maximum(
            _vector(
                node.declare_parameter(
                    "b601_semantic_linear_deadband_m", [0.015, 0.020, 0.010]
                ).value,
                3,
                "b601_semantic_linear_deadband_m",
            ),
            0.0,
        )
        self.semantic_max_observation_displacement_m = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_observation_displacement_m",
                [0.35, 0.35, 0.35],
            ).value,
            3,
            "b601_semantic_max_observation_displacement_m",
        )
        self.semantic_angle_deadband_rad = np.deg2rad(
            np.maximum(
                _vector(
                    node.declare_parameter(
                        "b601_semantic_angle_deadband_deg", [5.0, 5.0, 7.0]
                    ).value,
                    3,
                    "b601_semantic_angle_deadband_deg",
                ),
                0.0,
            )
        )
        self.semantic_max_joint_offset = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_joint_offset_rad",
                [0.035, 0.040, 0.040, 0.030, 0.010, 0.040],
            ).value,
            6,
            "b601_semantic_max_joint_offset_rad",
        )
        self.semantic_max_step = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_step_rad",
                [0.004, 0.004, 0.004, 0.003, 0.002, 0.004],
            ).value,
            6,
            "b601_semantic_max_step_rad",
        )
        self.semantic_max_tracking_error = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_tracking_error_rad",
                [0.012, 0.018, 0.018, 0.012, 0.008, 0.015],
            ).value,
            6,
            "b601_semantic_max_tracking_error_rad",
        )
        self.semantic_vertical_max_lift_m = float(
            node.declare_parameter(
                "b601_semantic_vertical_max_lift_m", 0.015
            ).value
        )
        if (
            not np.isfinite(self.semantic_vertical_max_lift_m)
            or self.semantic_vertical_max_lift_m < 0.0
        ):
            raise ValueError(
                "b601_semantic_vertical_max_lift_m must be finite and non-negative"
            )
        self.semantic_min_lift_before_articulation_m = float(
            node.declare_parameter(
                "b601_semantic_min_lift_before_articulation_m", 0.008
            ).value
        )
        if (
            not np.isfinite(self.semantic_min_lift_before_articulation_m)
            or self.semantic_min_lift_before_articulation_m < 0.0
        ):
            raise ValueError(
                "b601_semantic_min_lift_before_articulation_m must be finite "
                "and non-negative"
            )
        if (
            self.semantic_min_lift_before_articulation_m
            > self.semantic_vertical_max_lift_m
        ):
            raise ValueError(
                "b601_semantic_min_lift_before_articulation_m cannot exceed "
                "b601_semantic_vertical_max_lift_m"
            )
        self.semantic_min_chain_clearance_m = float(
            node.declare_parameter(
                "b601_semantic_min_chain_clearance_m", 0.001
            ).value
        )
        if (
            not np.isfinite(self.semantic_min_chain_clearance_m)
            or self.semantic_min_chain_clearance_m < 0.0
        ):
            raise ValueError(
                "b601_semantic_min_chain_clearance_m must be finite and non-negative"
            )
        self.semantic_vertical_joint_mask = _vector(
            node.declare_parameter(
                "b601_semantic_vertical_joint_mask",
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ).value,
            6,
            "b601_semantic_vertical_joint_mask",
        )
        if not np.all(np.isin(self.semantic_vertical_joint_mask, (0.0, 1.0))):
            raise ValueError("b601_semantic_vertical_joint_mask values must be 0 or 1")
        if not np.any(self.semantic_vertical_joint_mask):
            raise ValueError("b601_semantic_vertical_joint_mask cannot disable every joint")
        self.semantic_vertical_damping = float(
            node.declare_parameter("b601_semantic_vertical_damping", 0.02).value
        )
        self.semantic_vertical_orientation_weight = float(
            node.declare_parameter(
                "b601_semantic_vertical_orientation_weight_m_per_rad", 0.02
            ).value
        )
        self.semantic_vertical_height_profile = _positive_vector(
            node.declare_parameter(
                "b601_semantic_vertical_height_profile", [0.50, 0.82, 1.0]
            ).value,
            3,
            "b601_semantic_vertical_height_profile",
        )
        if (
            np.any(self.semantic_vertical_height_profile > 1.0)
            or np.any(np.diff(self.semantic_vertical_height_profile) < 0.0)
            or not np.isclose(self.semantic_vertical_height_profile[-1], 1.0)
        ):
            raise ValueError(
                "b601_semantic_vertical_height_profile must be non-decreasing, "
                "at most 1.0, and end at 1.0"
            )
        self.semantic_vertical_solver_step_rad = float(
            node.declare_parameter(
                "b601_semantic_vertical_solver_step_rad", 0.012
            ).value
        )
        self.semantic_vertical_solver_iterations = int(
            node.declare_parameter(
                "b601_semantic_vertical_solver_iterations", 8
            ).value
        )
        self.semantic_table_clearance_tolerance_m = float(
            node.declare_parameter(
                "b601_semantic_table_clearance_tolerance_m", 0.002
            ).value
        )
        if (
            not np.isfinite(self.semantic_vertical_damping)
            or self.semantic_vertical_damping <= 0.0
        ):
            raise ValueError("b601_semantic_vertical_damping must be positive")
        if (
            not np.isfinite(self.semantic_vertical_orientation_weight)
            or self.semantic_vertical_orientation_weight < 0.0
        ):
            raise ValueError(
                "b601_semantic_vertical_orientation_weight_m_per_rad cannot be negative"
            )
        if (
            not np.isfinite(self.semantic_vertical_solver_step_rad)
            or self.semantic_vertical_solver_step_rad <= 0.0
        ):
            raise ValueError(
                "b601_semantic_vertical_solver_step_rad must be positive and finite"
            )
        if not 1 <= self.semantic_vertical_solver_iterations <= 50:
            raise ValueError(
                "b601_semantic_vertical_solver_iterations must be between 1 and 50"
            )
        if (
            not np.isfinite(self.semantic_table_clearance_tolerance_m)
            or self.semantic_table_clearance_tolerance_m < 0.0
        ):
            raise ValueError(
                "b601_semantic_table_clearance_tolerance_m cannot be negative"
            )

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
        self.startup_joint_reference = None
        self.table_frame_heights = None
        self.semantic_control_reference_q = None
        self.semantic_reference_frame_heights = None
        self.semantic_control_side = None
        self.semantic_signal_references = {}
        self.semantic_signal_deltas = {
            "elbow": 0.0,
            "wrist": 0.0,
            "roll": 0.0,
        }
        self.last_semantic_log_time = 0.0
        self.last_semantic_vertical_error_m = 0.0
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
        import pinocchio as pin

        self._compute_fk = compute_fk
        self._compute_generalized_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._pin = pin
        self._ik_params = IKParams(
            max_iter=250,
            tolerance=1e-4,
            step_size=0.45,
            damping=1e-5,
        )
        self._model = load_robot_model(str(self.urdf_path.resolve()))
        self._model_data = self._model.createData()
        # Gravity feed-forward runs at 500 Hz on the SDK thread.  Pinocchio
        # Data is mutable, so visual IK/Jacobian work must use separate data.
        self._kinematics_data = self._model.createData()
        frame_id = self._model.getFrameId(self.end_effector_frame)
        if frame_id >= self._model.nframes:
            raise ValueError(
                f"B601 end-effector frame {self.end_effector_frame!r} is absent from the DM URDF"
            )
        self._end_frame_id = frame_id
        # Pinocchio's aligned-vector Python binding rejects numpy integer
        # scalars as indices.  Keep native ints while retaining list-based
        # advanced indexing for the NumPy height arrays below.
        self._articulation_clearance_frame_ids = resolve_native_frame_ids(
            self._model,
            ("link3", "link4", "end_link"),
        )
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
            outside_hard_limits = np.logical_or(
                q < self._joint_lower, q > self._joint_upper
            )
            if np.any(outside_hard_limits):
                raise StartupGuardError(
                    "Refusing to enable B601-DM torque: validated table feedback is "
                    "outside the URDF hard limits for joints "
                    f"{(np.flatnonzero(outside_hard_limits) + 1).tolist()}"
                )
            self.startup_joint_reference = q.copy()
            self._set_reference_from_q(q)
            self.table_frame_heights = self._frame_heights(q)
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
        self.semantic_control_reference_q = q.copy()
        self.semantic_reference_frame_heights = self._frame_heights(q)
        self.semantic_control_side = None
        self.semantic_signal_references.clear()
        self.semantic_signal_deltas.update(elbow=0.0, wrist=0.0, roll=0.0)
        self.last_semantic_vertical_error_m = 0.0
        self.last_target_pose = list(self.initial_pose)
        self.last_commanded_pose = list(self.initial_pose)
        self.last_arm_offset = np.zeros(3, dtype=np.float64)
        self.last_submit_time = 0.0
        self.first_tracking_motion_ok = False
        self.tracking_cycle_started_at = time.monotonic()
        self._reset_filters()
        self.logger.info(
            f"B601 {self.control_mode} control reference reset ({reason})"
        )
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
        self.semantic_control_side = None
        self.semantic_reference_frame_heights = None
        self.semantic_signal_references.clear()
        self.semantic_signal_deltas.update(elbow=0.0, wrist=0.0, roll=0.0)
        self._reset_filters()
        self.logger.info(
            f"B601 tracking paused; holding measured pose with gravity feed-forward ({reason})"
        )

    def update_from_operator_motion(
        self, observation: OperatorArmMotion | None
    ):
        """Update either the legacy Cartesian or semantic joint controller."""

        if observation is None:
            return None
        if self.control_mode == "cartesian":
            return self.update_from_palm_displacement(
                observation.palm_displacement_m
            )
        return self._update_semantic_joint_control(observation)

    @staticmethod
    def _scalar_deadband(value: float, threshold: float) -> float:
        magnitude = abs(float(value))
        threshold = max(0.0, float(threshold))
        if magnitude <= threshold:
            return 0.0
        return float(np.copysign(magnitude - threshold, value))

    def _reset_semantic_observation_reference(
        self,
        observation: OperatorArmMotion,
        q_reference: np.ndarray,
    ) -> None:
        self.semantic_control_side = observation.side
        self.semantic_control_reference_q = q_reference.copy()
        self.semantic_reference_frame_heights = self._frame_heights(q_reference)
        self.semantic_signal_references = {
            "elbow": observation.elbow_angle_rad,
            "wrist": observation.wrist_pitch_rad,
            "roll": observation.palm_roll_rad,
        }
        self.semantic_signal_deltas.update(elbow=0.0, wrist=0.0, roll=0.0)
        self._reset_filters()
        self.logger.info(
            "B601 semantic observation reference captured: "
            f"side={observation.side}, q={q_reference.round(4).tolist()}"
        )

    def _semantic_angle_delta(
        self,
        name: str,
        value: float | None,
        deadband_rad: float,
    ) -> float:
        if value is None or not np.isfinite(value):
            return float(self.semantic_signal_deltas[name])
        reference = self.semantic_signal_references.get(name)
        if reference is None or not np.isfinite(reference):
            self.semantic_signal_references[name] = float(value)
            self.semantic_signal_deltas[name] = 0.0
            return 0.0
        raw_delta = float(value) - float(reference)
        if name in ("wrist", "roll"):
            raw_delta = wrap_angle_radians(raw_delta)
        delta = self._scalar_deadband(raw_delta, deadband_rad)
        self.semantic_signal_deltas[name] = delta
        return delta

    def _update_semantic_joint_control(
        self,
        observation: OperatorArmMotion,
    ):
        if not self.enabled or self.initial_pose is None:
            return None
        if self.home_requested or self.home_motion_active:
            return self.last_arm_offset

        q_seed = (
            self.last_measured_q.copy()
            if self.last_measured_q is not None
            else self.target_q.copy()
        )
        if (
            self.semantic_control_reference_q is None
            or observation.side != self.semantic_control_side
        ):
            self._reset_semantic_observation_reference(observation, q_seed)
            return self.last_arm_offset

        now = time.monotonic()
        if now - self.last_submit_time < self.command_interval_s:
            return self.last_arm_offset

        camera_offset = _vector(
            observation.palm_displacement_m,
            3,
            "semantic palm displacement",
        )
        camera_offset *= self.palm_to_arm_scale
        if np.any(
            np.abs(camera_offset)
            > self.semantic_max_observation_displacement_m
        ):
            with self.target_lock:
                self.target_q = q_seed.copy()
            self._reset_filters()
            if now - self.last_semantic_log_time >= 1.0:
                self.logger.warn(
                    "Holding B601 measured pose: semantic palm displacement "
                    f"is implausible ({camera_offset.round(3).tolist()}m)"
                )
                self.last_semantic_log_time = now
            return self.last_arm_offset
        camera_offset = self._anti_drift_filter_offset(camera_offset)
        lateral = self._scalar_deadband(
            camera_offset[0], self.semantic_linear_deadband_m[0]
        )
        # Camera Z increases away from the camera; human forward motion is -Z.
        forward = self._scalar_deadband(
            -camera_offset[2], self.semantic_linear_deadband_m[1]
        )
        # The table pose is the lower boundary.  A hand moving down commands a
        # return toward zero lift, never motion below the validated table pose.
        lift = self._scalar_deadband(
            -camera_offset[1], self.semantic_linear_deadband_m[2]
        )
        lift = float(np.clip(lift, 0.0, self.semantic_vertical_max_lift_m))

        elbow_delta = self._semantic_angle_delta(
            "elbow",
            observation.elbow_angle_rad,
            self.semantic_angle_deadband_rad[0],
        )
        wrist_delta = self._semantic_angle_delta(
            "wrist",
            observation.wrist_pitch_rad,
            self.semantic_angle_deadband_rad[1],
        )
        roll_delta = self._semantic_angle_delta(
            "roll",
            observation.palm_roll_rad,
            self.semantic_angle_deadband_rad[2],
        )

        direct_offsets = semantic_joint_offsets(
            lateral,
            forward,
            elbow_delta,
            wrist_delta,
            roll_delta,
            self.semantic_joint_gains,
            self.semantic_channel_enable,
            self.semantic_max_joint_offset,
        )
        tcp_lift, chain_clearance = self._semantic_clearance_from_table(q_seed)
        articulation_ready = (
            tcp_lift >= self.semantic_min_lift_before_articulation_m
            and chain_clearance >= self.semantic_min_chain_clearance_m
            and lift >= self.semantic_min_lift_before_articulation_m
        )
        if not articulation_ready:
            direct_offsets[:] = 0.0
        q_reference = self.semantic_control_reference_q
        q_desired = q_reference + direct_offsets
        if self.semantic_channel_enable[5] > 0.5:
            q_desired = self._apply_semantic_vertical(q_desired, lift)
        else:
            self.last_semantic_vertical_error_m = 0.0

        lower, upper = joint_tracking_bounds(
            self._joint_lower,
            self._joint_upper,
            self.joint_limit_margin_rad,
            self.startup_joint_reference,
        )
        q_desired = np.clip(
            q_desired,
            q_reference - self.semantic_max_joint_offset,
            q_reference + self.semantic_max_joint_offset,
        )
        q_desired = np.clip(q_desired, lower, upper)
        with self.target_lock:
            previous_target = (
                q_seed.copy()
                if self.target_q is None
                else self.target_q.copy()
            )
        q_target = ramp_joint_target_with_tracking_window(
            q_desired,
            q_seed,
            previous_target,
            self.semantic_max_step,
            self.semantic_max_tracking_error,
        )
        if not self._semantic_table_clearance_ok(q_seed, q_target):
            if now - self.last_semantic_log_time >= 1.0:
                self.logger.warn(
                    "Refusing B601 semantic target: a modeled frame would move "
                    "below its validated table-pose height"
                )
                self.last_semantic_log_time = now
            return self.last_arm_offset
        if not self._joint_target_is_safe(q_target, q_seed):
            return self.last_arm_offset

        with self.target_lock:
            self.target_q = q_target
        self.last_submit_time = now
        self.last_arm_offset = np.array([lateral, forward, lift], dtype=np.float64)
        self.first_tracking_motion_ok = True
        if now - self.last_semantic_log_time >= 1.0:
            actual_vertical_error = self._semantic_vertical_error_at_q(
                q_seed,
                lift,
            )
            self.logger.info(
                "B601 semantic: "
                f"side={observation.side}, lateral={lateral:+.4f}m, "
                f"forward={forward:+.4f}m, lift={lift:+.4f}m, "
                f"tcp_lift={tcp_lift:+.4f}m, "
                f"chain_clearance={chain_clearance:+.4f}m, "
                f"solver_error={self.last_semantic_vertical_error_m:+.4f}m, "
                f"actual_vertical_error={actual_vertical_error:+.4f}m, "
                f"articulation={'ready' if articulation_ready else 'locked'}, "
                f"elbow={np.rad2deg(elbow_delta):+.1f}deg, "
                f"wrist={np.rad2deg(wrist_delta):+.1f}deg, "
                f"roll={np.rad2deg(roll_delta):+.1f}deg, "
                f"q_measured={q_seed.round(4).tolist()}, "
                f"q_desired={q_desired.round(4).tolist()}, "
                f"q_target={q_target.round(4).tolist()}, "
                f"target_lead={(q_target - q_seed).round(4).tolist()}"
            )
            self.last_semantic_log_time = now
        return self.last_arm_offset

    def _apply_semantic_vertical(
        self,
        q_base: np.ndarray,
        lift_m: float,
    ) -> np.ndarray:
        """Lift elbow, wrist and TCP together with a bounded DLS solve."""

        q = np.asarray(q_base, dtype=np.float64).copy()
        columns = np.flatnonzero(self.semantic_vertical_joint_mask > 0.5)
        if columns.size == 0:
            return q
        q_reference = self.semantic_control_reference_q
        reference_heights = self.semantic_reference_frame_heights
        if reference_heights is None:
            reference_heights = self._frame_heights(q_reference)
        target_heights = (
            reference_heights[self._articulation_clearance_frame_ids]
            + self.semantic_vertical_height_profile * float(lift_m)
        )
        tracking_lower, tracking_upper = joint_tracking_bounds(
            self._joint_lower,
            self._joint_upper,
            self.joint_limit_margin_rad,
            self.startup_joint_reference,
        )
        lower = np.maximum(
            q_reference - self.semantic_max_joint_offset,
            tracking_lower,
        )
        upper = np.minimum(
            q_reference + self.semantic_max_joint_offset,
            tracking_upper,
        )
        q = np.clip(q, lower, upper)

        height_error = np.zeros(3, dtype=np.float64)
        for _ in range(self.semantic_vertical_solver_iterations):
            q_model = self._pad_q_for_model(self._model, q, self.ARM_DOF)
            self._pin.forwardKinematics(
                self._model, self._kinematics_data, q_model
            )
            self._pin.updateFramePlacements(self._model, self._kinematics_data)
            current_heights = np.asarray(
                [
                    float(self._kinematics_data.oMf[frame_id].translation[2])
                    for frame_id in self._articulation_clearance_frame_ids
                ],
                dtype=np.float64,
            )
            height_error = target_heights - current_heights
            if float(np.max(np.abs(height_error))) < 2e-4:
                break
            height_rows = []
            end_jacobian = None
            for frame_id in self._articulation_clearance_frame_ids:
                jacobian = self._pin.computeFrameJacobian(
                    self._model,
                    self._kinematics_data,
                    q_model,
                    int(frame_id),
                    self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                height_rows.append(jacobian[2, columns])
                if int(frame_id) == self._end_frame_id:
                    end_jacobian = jacobian[:, columns]
            if end_jacobian is None:
                raise RuntimeError("B601 semantic vertical task is missing the TCP frame")
            task = np.vstack(
                (
                    np.asarray(height_rows, dtype=np.float64),
                    self.semantic_vertical_orientation_weight * end_jacobian[3:6, :],
                )
            )
            desired = np.concatenate(
                (height_error, np.zeros(3, dtype=np.float64))
            )
            regularized = task @ task.T + (
                self.semantic_vertical_damping**2
            ) * np.eye(task.shape[0])
            try:
                delta = task.T @ np.linalg.solve(regularized, desired)
            except np.linalg.LinAlgError:
                self.logger.warn("B601 semantic vertical Jacobian is singular; holding")
                return q_base.copy()
            # Element-wise clipping destroys the coordinated J2/J3/J4 ratio
            # and, in the near-straight table pose, can reverse TCP Z motion.
            delta = scale_vector_to_max_abs(
                delta,
                self.semantic_vertical_solver_step_rad,
            )
            proposed = q.copy()
            proposed[columns] += delta
            proposed = np.clip(proposed, lower, upper)
            if float(np.max(np.abs(proposed - q))) < 1e-8:
                break
            q = proposed

        final_heights = self._frame_heights(q)[
            self._articulation_clearance_frame_ids
        ]
        self.last_semantic_vertical_error_m = float(
            np.max(np.abs(target_heights - final_heights))
        )
        return q

    def _frame_heights(self, q: np.ndarray) -> np.ndarray:
        q_model = self._pad_q_for_model(self._model, q, self.ARM_DOF)
        self._pin.forwardKinematics(self._model, self._kinematics_data, q_model)
        self._pin.updateFramePlacements(self._model, self._kinematics_data)
        return np.asarray(
            [float(placement.translation[2]) for placement in self._kinematics_data.oMf],
            dtype=np.float64,
        )

    def _semantic_table_clearance_ok(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
    ) -> bool:
        reference = self.table_frame_heights
        if reference is None:
            return True
        current = np.asarray(current_q, dtype=np.float64)
        target = np.asarray(target_q, dtype=np.float64)
        for ratio in (0.5, 1.0):
            heights = self._frame_heights(current + ratio * (target - current))
            if heights.shape != reference.shape:
                return False
            if np.any(
                heights
                < reference - self.semantic_table_clearance_tolerance_m
            ):
                return False
        return True

    def _semantic_clearance_from_table(
        self, q: np.ndarray
    ) -> tuple[float, float]:
        if self.table_frame_heights is None:
            return 0.0, 0.0
        heights = self._frame_heights(q)
        tcp_lift = float(
            heights[self._end_frame_id]
            - self.table_frame_heights[self._end_frame_id]
        )
        chain_clearance = float(
            np.min(
                heights[self._articulation_clearance_frame_ids]
                - self.table_frame_heights[self._articulation_clearance_frame_ids]
            )
        )
        return tcp_lift, chain_clearance

    def _semantic_vertical_error_at_q(
        self,
        q: np.ndarray,
        lift_m: float,
    ) -> float:
        if self.semantic_reference_frame_heights is None:
            return 0.0
        target_heights = (
            self.semantic_reference_frame_heights[
                self._articulation_clearance_frame_ids
            ]
            + self.semantic_vertical_height_profile * float(lift_m)
        )
        actual_heights = self._frame_heights(q)[
            self._articulation_clearance_frame_ids
        ]
        return float(np.max(np.abs(target_heights - actual_heights)))

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
            self._kinematics_data,
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
        outside, jump = joint_target_safety_violations(
            target,
            current,
            self._joint_lower,
            self._joint_upper,
            self.joint_limit_margin_rad,
            self.max_joint_target_delta,
            self.startup_joint_reference,
        )
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
