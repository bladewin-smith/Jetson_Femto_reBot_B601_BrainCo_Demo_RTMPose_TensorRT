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
    semantic_direct_joint_offsets,
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


def weighted_damped_least_squares(
    jacobian: Sequence[Sequence[float]],
    error: Sequence[float],
    damping: float,
    joint_mobility: Sequence[float],
) -> np.ndarray:
    """Solve a task while weighting how much each controlled joint may move.

    ``joint_mobility`` is the diagonal of the joint-space mobility matrix.
    A larger value lets that joint contribute more to a redundant task; it is
    not a raw motor command or torque gain.
    """

    task = np.asarray(jacobian, dtype=np.float64)
    desired = np.asarray(error, dtype=np.float64).reshape(-1)
    mobility = np.asarray(joint_mobility, dtype=np.float64).reshape(-1)
    if task.ndim != 2 or task.shape[0] != desired.size:
        raise ValueError("Jacobian rows must match the task error size")
    if task.shape[1] != mobility.size:
        raise ValueError("Jacobian columns must match the joint mobility size")
    if not all(np.all(np.isfinite(value)) for value in (task, desired, mobility)):
        raise ValueError("Weighted DLS inputs must be finite")
    if np.any(mobility <= 0.0):
        raise ValueError("Joint mobility values must be positive")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("Weighted DLS damping must be positive and finite")

    mobility_matrix = np.diag(mobility)
    regularized = task @ mobility_matrix @ task.T + float(damping) ** 2 * np.eye(
        task.shape[0]
    )
    return mobility_matrix @ task.T @ np.linalg.solve(regularized, desired)


def horizontal_radial_axis(
    tcp_position_m: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Return a copied unit XY radial axis and the original radial distance.

    The explicit copy is safety-critical.  Pinocchio/NumPy position slices can
    be views; normalizing such a view in place would overwrite the reference
    TCP coordinates and create a fictitious forward error close to one metre.
    """

    position = _vector(tcp_position_m, 3, "tcp_position_m")
    axis = position[:2].copy()
    radius = float(np.linalg.norm(axis))
    if radius < 1e-6:
        raise ValueError("TCP is too close to the base axis for planar control")
    axis /= radius
    return axis, radius


def send_mit_checked(
    motor_map: dict[str, Any],
    joint_names: Sequence[str],
    position_rad: Sequence[float],
    velocity_rad_s: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    torque_nm: Sequence[float],
) -> None:
    """Send one MIT command per joint without hiding MotorBridge failures.

    The vendored SDK's ``JointGroup.send_mit`` catches ``CallError`` for each
    motor and silently continues. That behaviour is unsuitable for an
    unbraked arm: the target can look healthy in the outer loop while one
    member of a coordinated J2/J3/J4 motion receives no command. Keep sending
    the remaining motors so they retain their hold command, then report every
    failed joint to the caller.
    """

    names = [str(name) for name in joint_names]
    size = len(names)
    position = _vector(position_rad, size, "position_rad")
    velocity = _vector(velocity_rad_s, size, "velocity_rad_s")
    position_kp = _positive_vector(kp, size, "kp")
    velocity_kd = _positive_vector(kd, size, "kd")
    torque = _vector(torque_nm, size, "torque_nm")
    failures: list[str] = []
    for index, name in enumerate(names):
        if name not in motor_map:
            failures.append(f"{name}: missing MotorBridge handle")
            continue
        try:
            motor_map[name].send_mit(
                float(position[index]),
                float(velocity[index]),
                float(position_kp[index]),
                float(velocity_kd[index]),
                float(torque[index]),
            )
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("MIT send failed for " + "; ".join(failures))


def limit_mit_target_by_effort(
    measured_position_rad: Sequence[float],
    measured_velocity_rad_s: Sequence[float],
    requested_target_rad: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    feedforward_torque_nm: Sequence[float],
    effort_limit_nm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound the MIT target so estimated total joint effort stays limited.

    For MIT control the approximate commanded effort is
    ``kp * (q_target - q) - kd * dq + tau_ff``.  Limiting only gravity
    feed-forward is insufficient once a larger visible-motion position error
    is allowed.  This helper clips the position contribution while preserving
    the damping and gravity terms.
    """

    q = np.asarray(measured_position_rad, dtype=np.float64).reshape(-1)
    dq = np.asarray(measured_velocity_rad_s, dtype=np.float64).reshape(-1)
    requested = np.asarray(requested_target_rad, dtype=np.float64).reshape(-1)
    position_kp = np.asarray(kp, dtype=np.float64).reshape(-1)
    velocity_kd = np.asarray(kd, dtype=np.float64).reshape(-1)
    feedforward = np.asarray(feedforward_torque_nm, dtype=np.float64).reshape(-1)
    limit = np.asarray(effort_limit_nm, dtype=np.float64).reshape(-1)
    if not (
        q.shape
        == dq.shape
        == requested.shape
        == position_kp.shape
        == velocity_kd.shape
        == feedforward.shape
        == limit.shape
    ):
        raise ValueError("MIT effort-limit vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(values))
        for values in (q, dq, requested, position_kp, velocity_kd, feedforward, limit)
    ):
        raise ValueError("MIT effort-limit vectors must contain finite values")
    if np.any(position_kp <= 0.0) or np.any(limit <= 0.0):
        raise ValueError("MIT kp and effort limits must be positive")

    non_position_effort = feedforward - velocity_kd * dq
    requested_position_effort = position_kp * (requested - q)
    bounded_position_effort = np.clip(
        requested_position_effort,
        -limit - non_position_effort,
        limit - non_position_effort,
    )
    bounded_target = q + bounded_position_effort / position_kp
    estimated_effort = non_position_effort + bounded_position_effort
    limited = np.abs(bounded_target - requested) > 1e-9
    return bounded_target, estimated_effort, limited


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


def slew_limit_vector(
    requested: Sequence[float],
    previous: Sequence[float],
    max_velocity: Sequence[float],
    dt_s: float,
) -> np.ndarray:
    """Rate-limit a task-space command without changing its absolute bounds."""

    target = np.asarray(requested, dtype=np.float64).reshape(-1)
    prior = np.asarray(previous, dtype=np.float64).reshape(-1)
    velocity = np.asarray(max_velocity, dtype=np.float64).reshape(-1)
    dt = float(dt_s)
    if not target.shape == prior.shape == velocity.shape:
        raise ValueError("slew-limit vectors must have matching shapes")
    if not all(np.all(np.isfinite(value)) for value in (target, prior, velocity)):
        raise ValueError("slew-limit vectors must contain finite values")
    if np.any(velocity <= 0.0) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("slew-limit velocity and dt must be positive")
    maximum_step = velocity * dt
    return prior + np.clip(target - prior, -maximum_step, maximum_step)


def acceleration_limited_scalar_target(
    requested: float,
    previous: float,
    previous_velocity: float,
    max_velocity: float,
    max_acceleration: float,
    deadband: float,
    dt_s: float,
) -> tuple[float, float]:
    """Return a deadbanded velocity/acceleration-limited scalar target."""

    values = np.asarray(
        [
            requested,
            previous,
            previous_velocity,
            max_velocity,
            max_acceleration,
            deadband,
            dt_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("acceleration-limited target inputs must be finite")
    if max_velocity <= 0.0 or max_acceleration <= 0.0 or dt_s <= 0.0:
        raise ValueError("target velocity, acceleration and dt must be positive")
    if deadband < 0.0:
        raise ValueError("target deadband cannot be negative")

    error = float(requested - previous)
    if abs(error) <= deadband:
        return float(previous), 0.0
    requested_velocity = float(
        np.clip(error / dt_s, -max_velocity, max_velocity)
    )
    velocity_step = float(
        np.clip(
            requested_velocity - previous_velocity,
            -max_acceleration * dt_s,
            max_acceleration * dt_s,
        )
    )
    velocity = float(
        np.clip(previous_velocity + velocity_step, -max_velocity, max_velocity)
    )
    step = velocity * dt_s
    # Never cross the requested target. On arrival, remove residual velocity.
    if step * error > 0.0 and abs(step) >= abs(error):
        return float(requested), 0.0
    return float(previous + step), velocity


def filtered_wrapped_angle_delta(
    value: float,
    reference: float,
    previous_delta: float,
    deadband_rad: float,
    max_innovation_rad: float,
    filter_alpha: float,
) -> float:
    """Filter a periodic angle delta while bounding one-frame innovations.

    This is used by the independent J6 wrist-twist channel.  Wrapping both the
    reference delta and the innovation prevents a +/-pi boundary crossing from
    looking like a full revolution; innovation limiting and low-pass filtering
    prevent one bad depth frame from producing a visible J6 kick.
    """

    values = np.asarray(
        [
            value,
            reference,
            previous_delta,
            deadband_rad,
            max_innovation_rad,
            filter_alpha,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("wrapped-angle filter inputs must be finite")
    if deadband_rad < 0.0 or not 0.0 < filter_alpha <= 1.0:
        raise ValueError("angle deadband must be non-negative and alpha in (0, 1]")
    if not 0.0 < max_innovation_rad <= np.pi:
        raise ValueError("maximum angle innovation must be in (0, pi]")

    raw_delta = wrap_angle_radians(float(value) - float(reference))
    magnitude = abs(raw_delta)
    desired_delta = (
        0.0
        if magnitude <= deadband_rad
        else float(np.copysign(magnitude - deadband_rad, raw_delta))
    )
    innovation = wrap_angle_radians(desired_delta - float(previous_delta))
    limited_innovation = float(
        np.clip(innovation, -max_innovation_rad, max_innovation_rad)
    )
    return wrap_angle_radians(
        float(previous_delta) + filter_alpha * limited_innovation
    )


def confirm_directional_wrist_twist(
    filtered_delta_rad: float,
    active_direction: int,
    held_delta_rad: float,
    candidate_direction: int,
    candidate_count: int,
    activate_rad: float,
    release_rad: float,
    confirm_samples: int,
) -> tuple[int, float, int, int]:
    """Debounce wrist-twist direction before changing the J6 command.

    Depth-derived palm normals can briefly flip direction.  A confirmed
    direction is therefore held through isolated contradictory samples and is
    changed (or released to neutral) only after several consecutive samples.
    The downstream J6 velocity/acceleration limiter still shapes all motion.
    """

    values = np.asarray(
        [filtered_delta_rad, held_delta_rad, activate_rad, release_rad],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("wrist-twist confirmation inputs must be finite")
    if active_direction not in (-1, 0, 1) or candidate_direction not in (-1, 0, 1):
        raise ValueError("wrist-twist directions must be -1, 0, or 1")
    if confirm_samples < 1 or candidate_count < 0:
        raise ValueError("wrist-twist confirmation counts must be positive")
    if release_rad < 0.0 or activate_rad <= release_rad:
        raise ValueError("wrist-twist activation must exceed its release threshold")

    magnitude = abs(float(filtered_delta_rad))
    observed_direction = int(np.sign(filtered_delta_rad))
    if active_direction == 0:
        requested_direction = observed_direction if magnitude >= activate_rad else 0
    elif magnitude <= release_rad:
        requested_direction = 0
    elif observed_direction == active_direction:
        # A valid sample in the active direction refreshes the proportional
        # magnitude immediately and cancels any pending release/reversal.
        return active_direction, float(filtered_delta_rad), 0, 0
    elif magnitude >= activate_rad:
        requested_direction = observed_direction
    else:
        # Inside the Schmitt band, keep the last confirmed direction and value.
        return active_direction, float(held_delta_rad), 0, 0

    if requested_direction == active_direction:
        return active_direction, float(held_delta_rad), 0, 0
    if requested_direction != candidate_direction:
        candidate_direction = requested_direction
        candidate_count = 1
    else:
        candidate_count += 1
    if candidate_count < confirm_samples:
        return (
            active_direction,
            float(held_delta_rad),
            candidate_direction,
            candidate_count,
        )

    active_direction = candidate_direction
    held_delta_rad = (
        float(filtered_delta_rad) if active_direction != 0 else 0.0
    )
    return active_direction, held_delta_rad, 0, 0


def synchronize_vertical_follower_progress(
    reference_rad: Sequence[float],
    desired_rad: Sequence[float],
    measured_rad: Sequence[float],
    target_rad: Sequence[float],
    primary_index: int,
    follower_indices: Sequence[int],
    max_follower_progress_lead: float,
) -> tuple[np.ndarray, float, float]:
    """Keep fast lift joints from outrunning the load-bearing primary joint.

    Near the B601 table pose, J2 and J4 can track their MIT targets much faster
    than the heavily loaded J3.  Independent target windows then destroy the
    J2/J3/J4 ratio produced by the vertical solver and can lower the TCP.  This
    guard uses measured J3 trajectory progress as the clock and limits only the
    faster follower targets.  J3 retains its larger position-error allowance so
    it can generate enough torque to move.
    """

    reference = np.asarray(reference_rad, dtype=np.float64).reshape(-1)
    desired = np.asarray(desired_rad, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_rad, dtype=np.float64).reshape(-1)
    target = np.asarray(target_rad, dtype=np.float64).reshape(-1).copy()
    if not reference.shape == desired.shape == measured.shape == target.shape:
        raise ValueError("vertical synchronization vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(values))
        for values in (reference, desired, measured, target)
    ):
        raise ValueError("vertical synchronization vectors must contain finite values")
    primary = int(primary_index)
    if primary < 0 or primary >= reference.size:
        raise ValueError("vertical primary joint index is out of range")
    lead = float(max_follower_progress_lead)
    if not np.isfinite(lead) or lead < 0.0:
        raise ValueError("max_follower_progress_lead must be finite and non-negative")

    desired_delta = desired - reference
    primary_delta = float(desired_delta[primary])
    if abs(primary_delta) < 1e-8:
        return target, 1.0, 1.0
    primary_progress = float(
        np.clip(
            (measured[primary] - reference[primary]) / primary_delta,
            0.0,
            1.0,
        )
    )
    follower_progress_limit = float(np.clip(primary_progress + lead, 0.0, 1.0))
    for raw_index in follower_indices:
        index = int(raw_index)
        if index < 0 or index >= reference.size or index == primary:
            continue
        delta = float(desired_delta[index])
        if abs(delta) < 1e-8:
            continue
        progress_bound = reference[index] + follower_progress_limit * delta
        if delta > 0.0:
            target[index] = min(target[index], progress_bound)
        else:
            target[index] = max(target[index], progress_bound)
    return target, primary_progress, follower_progress_limit


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
                "b601_joint_kp", [120.0, 120.0, 120.0, 45.0, 35.0, 30.0]
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
        self.joint_effort_limit = _positive_vector(
            node.declare_parameter(
                "b601_joint_effort_limit_nm", [8.0, 10.0, 16.0, 6.0, 4.0, 4.0]
            ).value,
            self.ARM_DOF,
            "b601_joint_effort_limit_nm",
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
        # Channels: J1 palm lateral, J2/J3/J4 planar forward, two retired
        # compatibility slots, J6 palm roll, and J2/J3/J4 vertical motion.
        # Elbow and wrist observations no longer command individual joints.
        self.semantic_channel_enable = _vector(
            node.declare_parameter(
                "b601_semantic_channel_enable", [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
            ).value,
            6,
            "b601_semantic_channel_enable",
        )
        if not np.all(np.isin(self.semantic_channel_enable, (0.0, 1.0))):
            raise ValueError("b601_semantic_channel_enable values must be 0 or 1")
        self.semantic_joint_gains = _vector(
            node.declare_parameter(
                "b601_semantic_joint_gains",
                [0.60, -0.60, 0.20, 0.15, 0.45],
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
                        "b601_semantic_angle_deadband_deg", [5.0, 5.0, 3.0]
                    ).value,
                    3,
                    "b601_semantic_angle_deadband_deg",
                ),
                0.0,
            )
        )
        self.wrist_twist_filter_alpha = float(
            node.declare_parameter(
                "b601_wrist_twist_filter_alpha", 0.15
            ).value
        )
        self.wrist_twist_max_innovation_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_max_innovation_deg", 12.0
                ).value
            )
        )
        self.wrist_twist_activate_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_activate_deg", 12.0
                ).value
            )
        )
        self.wrist_twist_release_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_release_deg", 6.0
                ).value
            )
        )
        self.wrist_twist_confirm_samples = int(
            node.declare_parameter(
                "b601_wrist_twist_confirm_samples", 4
            ).value
        )
        if (
            not np.isfinite(self.wrist_twist_filter_alpha)
            or not 0.0 < self.wrist_twist_filter_alpha <= 1.0
            or not np.isfinite(self.wrist_twist_max_innovation_rad)
            or not 0.0 < self.wrist_twist_max_innovation_rad <= np.pi
            or not np.isfinite(self.wrist_twist_activate_rad)
            or not np.isfinite(self.wrist_twist_release_rad)
            or self.wrist_twist_release_rad < 0.0
            or self.wrist_twist_activate_rad <= self.wrist_twist_release_rad
            or self.wrist_twist_confirm_samples < 1
        ):
            raise ValueError(
                "B601 wrist-twist filter alpha must be in (0, 1] and "
                "maximum innovation in (0, 180] degrees; activation must "
                "exceed release and confirmation samples must be positive"
            )
        self.semantic_max_joint_offset = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_joint_offset_rad",
                [0.035, 0.040, 0.040, 0.030, 0.010, 0.300],
            ).value,
            6,
            "b601_semantic_max_joint_offset_rad",
        )
        self.wrist_twist_min_joint_offset_rad = float(
            node.declare_parameter(
                "b601_wrist_twist_min_joint_offset_rad", 0.140
            ).value
        )
        if (
            not np.isfinite(self.wrist_twist_min_joint_offset_rad)
            or self.wrist_twist_min_joint_offset_rad <= 0.0
            or self.wrist_twist_min_joint_offset_rad
            > self.semantic_max_joint_offset[5]
        ):
            raise ValueError(
                "B601 wrist-twist minimum J6 offset must be positive and no "
                "greater than the J6 maximum joint offset"
            )
        self.semantic_max_step = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_step_rad",
                [0.004, 0.004, 0.004, 0.003, 0.002, 0.006],
            ).value,
            6,
            "b601_semantic_max_step_rad",
        )
        self.semantic_max_tracking_error = _positive_vector(
            node.declare_parameter(
                "b601_semantic_max_tracking_error_rad",
                [0.012, 0.024, 0.040, 0.025, 0.008, 0.040],
            ).value,
            6,
            "b601_semantic_max_tracking_error_rad",
        )
        self.semantic_stall_error = _positive_vector(
            node.declare_parameter(
                "b601_semantic_stall_error_rad",
                [0.010, 0.015, 0.025, 0.015, 0.010, 0.012],
            ).value,
            6,
            "b601_semantic_stall_error_rad",
        )
        self.semantic_stall_max_velocity_rad_s = float(
            node.declare_parameter(
                "b601_semantic_stall_max_velocity_rad_s", 0.003
            ).value
        )
        # Retained as a declared compatibility parameter for existing launch
        # commands. Stall detection below is progress-based because a heavily
        # loaded J3 can move safely below this instantaneous velocity.
        self.semantic_stall_min_progress_rad = float(
            node.declare_parameter(
                "b601_semantic_stall_min_progress_rad", 0.0015
            ).value
        )
        self.semantic_stall_timeout_s = float(
            node.declare_parameter("b601_semantic_stall_timeout_s", 3.0).value
        )
        if (
            not np.isfinite(self.semantic_stall_max_velocity_rad_s)
            or self.semantic_stall_max_velocity_rad_s < 0.0
        ):
            raise ValueError(
                "b601_semantic_stall_max_velocity_rad_s must be finite and non-negative"
            )
        if (
            not np.isfinite(self.semantic_stall_min_progress_rad)
            or self.semantic_stall_min_progress_rad <= 0.0
        ):
            raise ValueError(
                "b601_semantic_stall_min_progress_rad must be positive and finite"
            )
        if (
            not np.isfinite(self.semantic_stall_timeout_s)
            or self.semantic_stall_timeout_s <= 0.0
        ):
            raise ValueError("b601_semantic_stall_timeout_s must be positive and finite")
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
        self.semantic_vertical_primary_joint = int(
            node.declare_parameter("b601_semantic_vertical_primary_joint", 3).value
        ) - 1
        vertical_indices = np.flatnonzero(self.semantic_vertical_joint_mask > 0.5)
        if not np.array_equal(vertical_indices, np.asarray([1, 2, 3])):
            raise ValueError(
                "b601_semantic_vertical_joint_mask must select exactly J2/J3/J4 "
                "for planar palm control"
            )
        if self.semantic_vertical_primary_joint not in vertical_indices:
            raise ValueError(
                "b601_semantic_vertical_primary_joint must select an enabled vertical joint"
            )
        self.semantic_vertical_follower_indices = np.asarray(
            [
                index
                for index in vertical_indices
                if index != self.semantic_vertical_primary_joint
            ],
            dtype=np.int64,
        )
        self.semantic_vertical_follower_progress_lead = float(
            node.declare_parameter(
                "b601_semantic_vertical_follower_progress_lead", 0.15
            ).value
        )
        if (
            not np.isfinite(self.semantic_vertical_follower_progress_lead)
            or self.semantic_vertical_follower_progress_lead < 0.0
            or self.semantic_vertical_follower_progress_lead > 1.0
        ):
            raise ValueError(
                "b601_semantic_vertical_follower_progress_lead must be in [0, 1]"
            )
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
        self.semantic_planar_input_scale = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_input_scale", [0.35, 1.0]
            ).value,
            2,
            "b601_semantic_planar_input_scale",
        )
        self.semantic_planar_max_forward_m = float(
            node.declare_parameter(
                "b601_semantic_planar_max_forward_m", 0.035
            ).value
        )
        if (
            not np.isfinite(self.semantic_planar_max_forward_m)
            or self.semantic_planar_max_forward_m <= 0.0
        ):
            raise ValueError(
                "b601_semantic_planar_max_forward_m must be positive and finite"
            )
        self.semantic_planar_joint_mobility = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_joint_mobility", [0.80, 1.0, 0.70]
            ).value,
            3,
            "b601_semantic_planar_joint_mobility",
        )
        # [TCP forward, link3 height, link4 height, TCP height].  These are
        # task weights, not motor gains or torque multipliers.
        self.semantic_planar_task_weights = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_task_weights", [1.0, 0.65, 0.85, 1.0]
            ).value,
            4,
            "b601_semantic_planar_task_weights",
        )
        self.semantic_planar_command_max_velocity = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_command_max_velocity_m_s", [0.040, 0.035]
            ).value,
            2,
            "b601_semantic_planar_command_max_velocity_m_s",
        )
        self.semantic_j4_target_max_velocity = float(
            node.declare_parameter(
                "b601_semantic_j4_target_max_velocity_rad_s", 0.025
            ).value
        )
        self.semantic_j4_target_max_acceleration = float(
            node.declare_parameter(
                "b601_semantic_j4_target_max_acceleration_rad_s2", 0.10
            ).value
        )
        self.semantic_j4_target_deadband = float(
            node.declare_parameter(
                "b601_semantic_j4_target_deadband_rad", 0.0015
            ).value
        )
        if (
            not np.isfinite(self.semantic_j4_target_max_velocity)
            or self.semantic_j4_target_max_velocity <= 0.0
            or not np.isfinite(self.semantic_j4_target_max_acceleration)
            or self.semantic_j4_target_max_acceleration <= 0.0
            or not np.isfinite(self.semantic_j4_target_deadband)
            or self.semantic_j4_target_deadband < 0.0
        ):
            raise ValueError(
                "B601 J4 target velocity/acceleration must be positive and "
                "deadband must be finite and non-negative"
            )
        self.semantic_j6_target_max_velocity = float(
            node.declare_parameter(
                "b601_semantic_j6_target_max_velocity_rad_s", 0.10
            ).value
        )
        self.semantic_j6_target_max_acceleration = float(
            node.declare_parameter(
                "b601_semantic_j6_target_max_acceleration_rad_s2", 0.50
            ).value
        )
        self.semantic_j6_target_deadband = float(
            node.declare_parameter(
                "b601_semantic_j6_target_deadband_rad", 0.0015
            ).value
        )
        if (
            not np.isfinite(self.semantic_j6_target_max_velocity)
            or self.semantic_j6_target_max_velocity <= 0.0
            or not np.isfinite(self.semantic_j6_target_max_acceleration)
            or self.semantic_j6_target_max_acceleration <= 0.0
            or not np.isfinite(self.semantic_j6_target_deadband)
            or self.semantic_j6_target_deadband < 0.0
        ):
            raise ValueError(
                "B601 J6 target velocity/acceleration must be positive and "
                "deadband must be finite and non-negative"
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
        self.last_measured_tau = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.last_gravity_tau = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.last_sent_target_q = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.last_command_effort = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.last_effort_limited = np.zeros(self.ARM_DOF, dtype=bool)
        self.last_motor_status = np.zeros(self.ARM_DOF, dtype=np.int64)
        self.semantic_stall_started_s = np.full(self.ARM_DOF, np.nan, dtype=np.float64)
        self.semantic_stall_reference_q = np.full(self.ARM_DOF, np.nan, dtype=np.float64)
        self.semantic_stall_direction = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.semantic_actuation_fault = None
        self.startup_joint_reference = None
        self.table_frame_heights = None
        self.semantic_control_reference_q = None
        self.semantic_reference_frame_heights = None
        self.semantic_control_side = None
        self.semantic_signal_references = {}
        self.semantic_signal_deltas = {
            "wrist_twist": 0.0,
        }
        self.semantic_j6_clearance_enabled = False
        self.semantic_j6_active_direction = 0
        self.semantic_j6_held_delta_rad = 0.0
        self.semantic_j6_candidate_direction = 0
        self.semantic_j6_candidate_count = 0
        self.semantic_j6_command_offset_rad = 0.0
        self.last_semantic_log_time = 0.0
        self.last_semantic_vertical_error_m = 0.0
        self.last_semantic_planar_error_m = np.zeros(2, dtype=np.float64)
        self.last_semantic_forward_gate = 0.0
        self.semantic_planar_command_m = np.zeros(2, dtype=np.float64)
        self.last_semantic_planar_command_time = 0.0
        self.semantic_j4_target_velocity = 0.0
        self.semantic_j6_target_velocity = 0.0
        self.last_semantic_vertical_primary_progress = 0.0
        self.last_semantic_vertical_follower_limit = 0.0
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
                states = [
                    robot._motor_map[name].get_state()
                    for name in robot.arm.joint_names
                ]
                missing = [
                    robot.arm.joint_names[index]
                    for index, state in enumerate(states)
                    if state is None
                ]
                if missing:
                    raise RuntimeError(f"missing MotorBridge feedback from {missing}")
                status = np.asarray(
                    [int(getattr(state, "status_code", -1)) for state in states],
                    dtype=np.int64,
                )
                disabled_or_faulted = [
                    f"{robot.arm.joint_names[index]}=0x{int(code):X}"
                    for index, code in enumerate(status)
                    if int(code) != 0x1
                ]
                if disabled_or_faulted:
                    raise RuntimeError(
                        "B601 motor is not ENABLED: " + ", ".join(disabled_or_faulted)
                    )
                measured_tau = np.asarray(
                    [float(getattr(state, "torq", 0.0)) for state in states],
                    dtype=np.float64,
                )
                with self.target_lock:
                    target_q = q.copy() if self.target_q is None else self.target_q.copy()
                tau = self._gravity_torque(q)
                sent_target_q, command_effort, effort_limited = limit_mit_target_by_effort(
                    q,
                    dq,
                    target_q,
                    self.joint_kp,
                    self.joint_kd,
                    tau,
                    self.joint_effort_limit,
                )
                send_mit_checked(
                    robot._motor_map,
                    robot.arm.joint_names,
                    sent_target_q,
                    np.zeros(self.ARM_DOF),
                    self.joint_kp,
                    self.joint_kd,
                    tau,
                )
            self.last_measured_q = q
            self.last_measured_dq = dq
            self.last_measured_tau = measured_tau
            self.last_gravity_tau = tau
            self.last_sent_target_q = sent_target_q
            self.last_command_effort = command_effort
            self.last_effort_limited = effort_limited
            self.last_motor_status = status
            self._update_semantic_stall_watchdog(q, dq, sent_target_q)
            self.last_control_error = None
        except Exception as exc:
            self.last_control_error = exc
            now = time.monotonic()
            if now - self.last_control_error_log_s >= self.sdk_busy_log_interval_s:
                self.logger.error(
                    f"B601 control-loop error; the last target remains active: {exc}"
                )
                self.last_control_error_log_s = now

    def _update_semantic_stall_watchdog(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        target_q: np.ndarray,
    ) -> None:
        """Latch a tracking fault when a commanded joint remains stationary."""

        if self.control_mode != "semantic_joint" or self.semantic_control_reference_q is None:
            self.semantic_stall_started_s.fill(np.nan)
            self.semantic_stall_reference_q.fill(np.nan)
            self.semantic_stall_direction.fill(0.0)
            return
        now = time.monotonic()
        signed_error = np.asarray(target_q) - np.asarray(q)
        demanding = np.abs(signed_error) >= self.semantic_stall_error
        inactive = ~demanding
        self.semantic_stall_started_s[inactive] = np.nan
        self.semantic_stall_reference_q[inactive] = np.nan
        self.semantic_stall_direction[inactive] = 0.0
        for index in np.flatnonzero(demanding):
            direction = float(np.sign(signed_error[index]))
            uninitialized = np.isnan(self.semantic_stall_started_s[index])
            direction_changed = direction != self.semantic_stall_direction[index]
            if uninitialized or direction_changed:
                self.semantic_stall_started_s[index] = now
                self.semantic_stall_reference_q[index] = q[index]
                self.semantic_stall_direction[index] = direction
                continue
            progress = direction * (q[index] - self.semantic_stall_reference_q[index])
            if progress >= self.semantic_stall_min_progress_rad:
                self.semantic_stall_started_s[index] = now
                self.semantic_stall_reference_q[index] = q[index]
        timed_out = np.logical_and(
            demanding,
            now - self.semantic_stall_started_s >= self.semantic_stall_timeout_s,
        )
        if not np.any(timed_out) or self.semantic_actuation_fault is not None:
            return
        indices = np.flatnonzero(timed_out)
        details = []
        for index in indices:
            progress_since_reset = self.semantic_stall_direction[index] * (
                q[index] - self.semantic_stall_reference_q[index]
            )
            details.append(
                f"J{index + 1}(error={target_q[index] - q[index]:+.4f}rad, "
                f"progress={progress_since_reset:+.4f}rad, "
                f"dq={dq[index]:+.4f}rad/s, "
                f"status=0x{int(self.last_motor_status[index]):X})"
            )
        self.semantic_actuation_fault = (
            "semantic joint tracking stalled: " + ", ".join(details)
        )
        with self.target_lock:
            self.target_q = np.asarray(q, dtype=np.float64).copy()
        self.logger.error(
            "B601 actuation watchdog stopped this tracking cycle; holding the "
            f"measured pose. {self.semantic_actuation_fault}. Press 's' only after "
            "checking the named motor, its connector and the 24 V supply."
        )

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

    def _reset_wrist_twist_confirmation(self) -> None:
        self.semantic_j6_clearance_enabled = False
        self.semantic_j6_active_direction = 0
        self.semantic_j6_held_delta_rad = 0.0
        self.semantic_j6_candidate_direction = 0
        self.semantic_j6_candidate_count = 0
        self.semantic_j6_command_offset_rad = 0.0

    def _reset_semantic_smoothing(self, now_s: float = 0.0) -> None:
        self.semantic_planar_command_m.fill(0.0)
        self.last_semantic_planar_command_time = float(now_s)
        self.semantic_j4_target_velocity = 0.0
        self.semantic_j6_target_velocity = 0.0
        self._reset_wrist_twist_confirmation()

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
        self.semantic_signal_deltas.update(wrist_twist=0.0)
        self.last_semantic_vertical_error_m = 0.0
        self.last_semantic_planar_error_m.fill(0.0)
        self.last_semantic_forward_gate = 0.0
        self._reset_semantic_smoothing(time.monotonic())
        self.last_target_pose = list(self.initial_pose)
        self.last_commanded_pose = list(self.initial_pose)
        self.last_arm_offset = np.zeros(3, dtype=np.float64)
        self.last_submit_time = 0.0
        self.first_tracking_motion_ok = False
        self.tracking_cycle_started_at = time.monotonic()
        self.semantic_stall_started_s.fill(np.nan)
        self.semantic_stall_reference_q.fill(np.nan)
        self.semantic_stall_direction.fill(0.0)
        self.semantic_actuation_fault = None
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
        self.semantic_signal_deltas.update(wrist_twist=0.0)
        self.last_semantic_planar_error_m.fill(0.0)
        self.last_semantic_forward_gate = 0.0
        self._reset_semantic_smoothing()
        self.semantic_stall_started_s.fill(np.nan)
        self.semantic_stall_reference_q.fill(np.nan)
        self.semantic_stall_direction.fill(0.0)
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
            "wrist_twist": observation.palm_roll_rad,
        }
        self.semantic_signal_deltas.update(wrist_twist=0.0)
        self._reset_semantic_smoothing(time.monotonic())
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
        if name == "wrist_twist":
            delta = filtered_wrapped_angle_delta(
                float(value),
                float(reference),
                float(self.semantic_signal_deltas[name]),
                deadband_rad,
                self.wrist_twist_max_innovation_rad,
                self.wrist_twist_filter_alpha,
            )
        else:
            delta = self._scalar_deadband(
                float(value) - float(reference), deadband_rad
            )
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
        if self.semantic_actuation_fault is not None:
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
            self.last_semantic_planar_command_time = now
            self.semantic_j4_target_velocity = 0.0
            self.semantic_j6_target_velocity = 0.0
            self._reset_wrist_twist_confirmation()
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
        forward *= self.semantic_planar_input_scale[0]
        forward = float(
            np.clip(
                forward,
                -self.semantic_planar_max_forward_m,
                self.semantic_planar_max_forward_m,
            )
        )
        # The table pose is the lower boundary.  A hand moving down commands a
        # return toward zero lift, never motion below the validated table pose.
        lift = self._scalar_deadband(
            -camera_offset[1], self.semantic_linear_deadband_m[2]
        )
        lift *= self.semantic_planar_input_scale[1]
        lift = float(np.clip(lift, 0.0, self.semantic_vertical_max_lift_m))

        tcp_lift, chain_clearance = self._semantic_clearance_from_table(q_seed)
        tcp_fraction = (
            1.0
            if self.semantic_min_lift_before_articulation_m <= 0.0
            else float(
                np.clip(
                    tcp_lift / self.semantic_min_lift_before_articulation_m,
                    0.0,
                    1.0,
                )
            )
        )
        chain_fraction = (
            1.0
            if self.semantic_min_chain_clearance_m <= 0.0
            else float(
                np.clip(
                    chain_clearance / self.semantic_min_chain_clearance_m,
                    0.0,
                    1.0,
                )
            )
        )
        clearance_fraction = min(tcp_fraction, chain_fraction)
        # Smoothstep avoids a visible target discontinuity at the old hard
        # articulation threshold.  J1 and the independently estimated J6
        # wrist twist remain safety-gated until the measured arm clears the
        # tabletop; their vision signals do not otherwise depend on the planar
        # J2/J3/J4 solver.
        forward_gate = clearance_fraction**2 * (3.0 - 2.0 * clearance_fraction)
        self.last_semantic_forward_gate = forward_gate
        articulation_ready = (
            clearance_fraction >= 1.0 - 1e-6
        )
        # Arm J6 only after the links have reached near-full clearance, then
        # capture a fresh twist zero.  Lifting the palm from the table changes
        # the human arm plane considerably and must not be mistaken for a
        # deliberate wrist rotation.  A lower disarm threshold prevents gate
        # chatter when measured clearance sits close to the boundary.
        if (
            not self.semantic_j6_clearance_enabled
            and clearance_fraction >= 0.95
            and observation.palm_roll_rad is not None
            and np.isfinite(observation.palm_roll_rad)
        ):
            self.semantic_j6_clearance_enabled = True
            self.semantic_signal_references["wrist_twist"] = float(
                observation.palm_roll_rad
            )
            self.semantic_signal_deltas["wrist_twist"] = 0.0
            self.semantic_j6_active_direction = 0
            self.semantic_j6_held_delta_rad = 0.0
            self.semantic_j6_candidate_direction = 0
            self.semantic_j6_candidate_count = 0
        elif self.semantic_j6_clearance_enabled and clearance_fraction <= 0.50:
            self._reset_wrist_twist_confirmation()

        roll_delta = self._semantic_angle_delta(
            "wrist_twist",
            observation.palm_roll_rad,
            self.semantic_angle_deadband_rad[2],
        )
        if self.semantic_j6_clearance_enabled:
            (
                self.semantic_j6_active_direction,
                self.semantic_j6_held_delta_rad,
                self.semantic_j6_candidate_direction,
                self.semantic_j6_candidate_count,
            ) = confirm_directional_wrist_twist(
                roll_delta,
                self.semantic_j6_active_direction,
                self.semantic_j6_held_delta_rad,
                self.semantic_j6_candidate_direction,
                self.semantic_j6_candidate_count,
                self.wrist_twist_activate_rad,
                self.wrist_twist_release_rad,
                self.wrist_twist_confirm_samples,
            )
        else:
            roll_delta = 0.0

        direct_offsets = semantic_direct_joint_offsets(
            lateral,
            self.semantic_j6_held_delta_rad,
            self.semantic_joint_gains,
            self.semantic_channel_enable,
            self.semantic_max_joint_offset,
        )
        if (
            self.semantic_channel_enable[4] > 0.5
            and self.semantic_j6_active_direction != 0
            and abs(float(direct_offsets[5])) > 1e-9
        ):
            direct_offsets[5] = float(
                np.copysign(
                    max(
                        abs(float(direct_offsets[5])),
                        self.wrist_twist_min_joint_offset_rad,
                    ),
                    float(direct_offsets[5]),
                )
            )
            direct_offsets[5] = float(
                np.clip(
                    direct_offsets[5],
                    -self.semantic_max_joint_offset[5],
                    self.semantic_max_joint_offset[5],
                )
            )
        else:
            direct_offsets[5] = 0.0
        self.semantic_j6_command_offset_rad = float(direct_offsets[5])
        direct_offsets[0] *= forward_gate
        direct_offsets[5] *= float(self.semantic_j6_clearance_enabled)
        q_reference = self.semantic_control_reference_q
        q_base = q_reference + direct_offsets
        requested_forward_target = (
            forward * forward_gate
            if self.semantic_channel_enable[1] > 0.5
            else 0.0
        )
        requested_lift_target = (
            lift if self.semantic_channel_enable[5] > 0.5 else 0.0
        )
        planar_dt = (
            self.command_interval_s
            if self.last_semantic_planar_command_time <= 0.0
            else float(
                np.clip(
                    now - self.last_semantic_planar_command_time,
                    0.001,
                    0.2,
                )
            )
        )
        planar_command = slew_limit_vector(
            [requested_forward_target, requested_lift_target],
            self.semantic_planar_command_m,
            self.semantic_planar_command_max_velocity,
            planar_dt,
        )
        forward_target = float(planar_command[0])
        lift_target = float(planar_command[1])
        if abs(forward_target) > 0.0 or lift_target > 0.0:
            q_desired = self._apply_semantic_planar(
                q_base,
                forward_target,
                lift_target,
            )
        else:
            q_desired = q_base
            self.last_semantic_vertical_error_m = 0.0
            self.last_semantic_planar_error_m.fill(0.0)

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
        if not articulation_ready and lift_target > 0.0:
            (
                q_target,
                self.last_semantic_vertical_primary_progress,
                self.last_semantic_vertical_follower_limit,
            ) = synchronize_vertical_follower_progress(
                q_reference,
                q_desired,
                q_seed,
                q_target,
                self.semantic_vertical_primary_joint,
                self.semantic_vertical_follower_indices,
                self.semantic_vertical_follower_progress_lead,
            )
        else:
            self.last_semantic_vertical_primary_progress = 1.0
            self.last_semantic_vertical_follower_limit = 1.0
        target_dt = (
            self.command_interval_s
            if self.last_submit_time <= 0.0
            else float(np.clip(now - self.last_submit_time, 0.001, 0.2))
        )
        j4_target, j4_velocity = acceleration_limited_scalar_target(
            float(q_target[3]),
            float(previous_target[3]),
            self.semantic_j4_target_velocity,
            self.semantic_j4_target_max_velocity,
            self.semantic_j4_target_max_acceleration,
            self.semantic_j4_target_deadband,
            target_dt,
        )
        q_target[3] = j4_target
        j6_target, j6_velocity = acceleration_limited_scalar_target(
            float(q_target[5]),
            float(previous_target[5]),
            self.semantic_j6_target_velocity,
            self.semantic_j6_target_max_velocity,
            self.semantic_j6_target_max_acceleration,
            self.semantic_j6_target_deadband,
            target_dt,
        )
        q_target[5] = j6_target
        if not self._semantic_table_clearance_ok(q_seed, q_target):
            self.last_semantic_planar_command_time = now
            self.semantic_j4_target_velocity = 0.0
            self.semantic_j6_target_velocity = 0.0
            if now - self.last_semantic_log_time >= 1.0:
                self.logger.warn(
                    "Refusing B601 semantic target: a modeled frame would move "
                    "below its validated table-pose height"
                )
                self.last_semantic_log_time = now
            return self.last_arm_offset
        if not self._joint_target_is_safe(q_target, q_seed):
            self.last_semantic_planar_command_time = now
            self.semantic_j4_target_velocity = 0.0
            self.semantic_j6_target_velocity = 0.0
            return self.last_arm_offset

        with self.target_lock:
            self.target_q = q_target
        self.semantic_planar_command_m = planar_command
        self.last_semantic_planar_command_time = now
        self.semantic_j4_target_velocity = j4_velocity
        self.semantic_j6_target_velocity = j6_velocity
        self.last_submit_time = now
        self.last_arm_offset = np.array([lateral, forward, lift], dtype=np.float64)
        self.first_tracking_motion_ok = True
        if now - self.last_semantic_log_time >= 1.0:
            wrist_twist_text = (
                "invalid"
                if observation.palm_roll_rad is None
                else f"{np.rad2deg(observation.palm_roll_rad):+.1f}deg"
            )
            actual_planar_error = self._semantic_planar_error_at_q(
                q_seed,
                q_base,
                forward_target,
                lift_target,
            )
            self.logger.info(
                "B601 semantic: "
                f"side={observation.side}, lateral={lateral:+.4f}m, "
                f"forward={forward:+.4f}m, planar_forward={forward_target:+.4f}m, "
                f"lift={lift_target:+.4f}m, forward_gate={forward_gate:.2f}, "
                f"tcp_lift={tcp_lift:+.4f}m, "
                f"chain_clearance={chain_clearance:+.4f}m, "
                f"solver_error=[forward={self.last_semantic_planar_error_m[0]:+.4f}, "
                f"height={self.last_semantic_planar_error_m[1]:+.4f}]m, "
                f"actual_error=[forward={actual_planar_error[0]:+.4f}, "
                f"height={actual_planar_error[1]:+.4f}]m, "
                f"J3_progress={self.last_semantic_vertical_primary_progress:.2f}, "
                f"follower_limit={self.last_semantic_vertical_follower_limit:.2f}, "
                f"J4_target_velocity={self.semantic_j4_target_velocity:+.4f}rad/s, "
                f"J6_target_velocity={self.semantic_j6_target_velocity:+.4f}rad/s, "
                f"articulation={'ready' if articulation_ready else 'locked'}, "
                f"J6_clearance={'armed' if self.semantic_j6_clearance_enabled else 'locked'}, "
                f"wrist_twist={wrist_twist_text}, "
                f"J6_delta={np.rad2deg(roll_delta):+.1f}deg, "
                f"J6_confirmed={np.rad2deg(self.semantic_j6_held_delta_rad):+.1f}deg, "
                f"J6_direction={self.semantic_j6_active_direction:+d}, "
                f"J6_pending={self.semantic_j6_candidate_count}/{self.wrist_twist_confirm_samples}, "
                f"J6_offset={np.rad2deg(self.semantic_j6_command_offset_rad):+.1f}deg, "
                f"J6_measured_offset={np.rad2deg(q_seed[5] - q_reference[5]):+.1f}deg, "
                f"q_measured={q_seed.round(4).tolist()}, "
                f"q_desired={q_desired.round(4).tolist()}, "
                f"q_target={q_target.round(4).tolist()}, "
                f"target_lead={(q_target - q_seed).round(4).tolist()}, "
                f"q_sent={self.last_sent_target_q.round(4).tolist()}, "
                f"tau_g={self.last_gravity_tau.round(3).tolist()}, "
                f"tau_measured={self.last_measured_tau.round(3).tolist()}, "
                f"effort_cmd={self.last_command_effort.round(3).tolist()}, "
                f"effort_limited={self.last_effort_limited.tolist()}, "
                f"motor_status={[f'0x{int(code):X}' for code in self.last_motor_status]}"
            )
            self.last_semantic_log_time = now
        return self.last_arm_offset

    def _apply_semantic_planar(
        self,
        q_base: np.ndarray,
        forward_m: float,
        lift_m: float,
    ) -> np.ndarray:
        """Solve palm forward/height together over J2/J3/J4.

        The forward axis is the current horizontal base-to-TCP radial axis.
        This makes the task independent of J1's yaw.  Three link-height tasks
        retain the tabletop-safe lift profile while a weighted joint-space
        metric determines how J2/J3/J4 share the requested motion.
        """

        q = np.asarray(q_base, dtype=np.float64).copy()
        columns = np.flatnonzero(self.semantic_vertical_joint_mask > 0.5)
        if columns.size == 0:
            return q
        q_reference = self.semantic_control_reference_q
        base_positions = self._frame_positions(q_base)
        base_tcp = base_positions[self._end_frame_id]
        try:
            forward_axis, base_forward = horizontal_radial_axis(base_tcp)
        except ValueError:
            self.logger.warn(
                "B601 TCP is too close to the base axis for planar control; holding"
            )
            return q
        target_forward = base_forward + float(forward_m)
        reference_heights = base_positions[:, 2]
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

        forward_error = 0.0
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
            current_tcp = np.asarray(
                self._kinematics_data.oMf[self._end_frame_id].translation,
                dtype=np.float64,
            )
            forward_error = target_forward - float(
                np.dot(current_tcp[:2], forward_axis)
            )
            height_error = target_heights - current_heights
            if max(abs(forward_error), float(np.max(np.abs(height_error)))) < 2e-4:
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
                raise RuntimeError("B601 semantic planar task is missing the TCP frame")
            forward_row = (
                forward_axis[0] * end_jacobian[0, :]
                + forward_axis[1] * end_jacobian[1, :]
            )
            position_task = np.vstack(
                (forward_row, np.asarray(height_rows, dtype=np.float64))
            )
            weighted_position_task = (
                self.semantic_planar_task_weights[:, None] * position_task
            )
            task = np.vstack(
                (
                    weighted_position_task,
                    self.semantic_vertical_orientation_weight * end_jacobian[3:6, :],
                )
            )
            desired = np.concatenate(
                (
                    self.semantic_planar_task_weights
                    * np.concatenate(([forward_error], height_error)),
                    np.zeros(3, dtype=np.float64),
                )
            )
            try:
                delta = weighted_damped_least_squares(
                    task,
                    desired,
                    self.semantic_vertical_damping,
                    self.semantic_planar_joint_mobility,
                )
            except np.linalg.LinAlgError:
                self.logger.warn("B601 semantic planar Jacobian is singular; holding")
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

        final_positions = self._frame_positions(q)
        final_forward = float(
            np.dot(final_positions[self._end_frame_id, :2], forward_axis)
        )
        final_height_error = target_heights - final_positions[
            self._articulation_clearance_frame_ids, 2
        ]
        self.last_semantic_planar_error_m = np.asarray(
            [
                target_forward - final_forward,
                float(np.max(np.abs(final_height_error))),
            ],
            dtype=np.float64,
        )
        self.last_semantic_vertical_error_m = self.last_semantic_planar_error_m[1]
        return q

    def _frame_positions(self, q: np.ndarray) -> np.ndarray:
        q_model = self._pad_q_for_model(self._model, q, self.ARM_DOF)
        self._pin.forwardKinematics(self._model, self._kinematics_data, q_model)
        self._pin.updateFramePlacements(self._model, self._kinematics_data)
        return np.asarray(
            [placement.translation for placement in self._kinematics_data.oMf],
            dtype=np.float64,
        )

    def _frame_heights(self, q: np.ndarray) -> np.ndarray:
        return self._frame_positions(q)[:, 2]

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
        current_heights = self._frame_heights(current)
        if current_heights.shape != reference.shape:
            return False
        # If tracking mismatch has already put a modeled frame just below the
        # table tolerance, permit only a monotonic-height recovery.  Requiring
        # every recovery sample to be above the original floor would otherwise
        # trap the arm at the low pose and repeatedly reject the corrective
        # J2/J3/J4 command.
        recovery_floor = np.minimum(
            reference - self.semantic_table_clearance_tolerance_m,
            current_heights,
        )
        for ratio in (0.5, 1.0):
            heights = self._frame_heights(current + ratio * (target - current))
            if heights.shape != reference.shape:
                return False
            if np.any(heights < recovery_floor - 1e-6):
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

    def _semantic_planar_error_at_q(
        self,
        q: np.ndarray,
        q_base: np.ndarray,
        forward_m: float,
        lift_m: float,
    ) -> np.ndarray:
        base_positions = self._frame_positions(q_base)
        actual_positions = self._frame_positions(q)
        base_tcp = base_positions[self._end_frame_id]
        try:
            forward_axis, base_forward = horizontal_radial_axis(base_tcp)
        except ValueError:
            return np.zeros(2, dtype=np.float64)
        target_forward = base_forward + float(forward_m)
        actual_forward = float(
            np.dot(actual_positions[self._end_frame_id, :2], forward_axis)
        )
        target_heights = (
            base_positions[self._articulation_clearance_frame_ids, 2]
            + self.semantic_vertical_height_profile * float(lift_m)
        )
        actual_heights = actual_positions[
            self._articulation_clearance_frame_ids, 2
        ]
        return np.asarray(
            [
                target_forward - actual_forward,
                float(np.max(np.abs(target_heights - actual_heights))),
            ],
            dtype=np.float64,
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
        self._reset_semantic_smoothing()
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
