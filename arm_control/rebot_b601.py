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


def intersect_joint_position_limits(
    model_lower_rad: Sequence[float],
    model_upper_rad: Sequence[float],
    safety_lower_rad: Sequence[float],
    safety_upper_rad: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect model limits with an independently configured safety envelope.

    Seeed's published B601-DM range table and the current official DM URDF are
    not identical.  Treating either source as permission to exceed the other
    would be unsafe, so runtime control uses their intersection.  The normal
    joint-limit margin is applied later by :func:`joint_tracking_bounds`.
    """

    model_lower = np.asarray(model_lower_rad, dtype=np.float64).reshape(-1)
    model_upper = np.asarray(model_upper_rad, dtype=np.float64).reshape(-1)
    safety_lower = np.asarray(safety_lower_rad, dtype=np.float64).reshape(-1)
    safety_upper = np.asarray(safety_upper_rad, dtype=np.float64).reshape(-1)
    if not (
        model_lower.shape
        == model_upper.shape
        == safety_lower.shape
        == safety_upper.shape
    ):
        raise ValueError("joint position-limit vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (model_lower, model_upper, safety_lower, safety_upper)
    ):
        raise ValueError("joint position limits must be finite")
    if np.any(model_lower >= model_upper) or np.any(safety_lower >= safety_upper):
        raise ValueError("joint position lower limits must be below upper limits")

    lower = np.maximum(model_lower, safety_lower)
    upper = np.minimum(model_upper, safety_upper)
    if np.any(lower >= upper):
        raise ValueError("model and configured joint safety limits do not overlap")
    return lower, upper


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
    coordinated_indices: Sequence[int] = (),
) -> np.ndarray:
    """Ramp from the prior target while bounding target lead over feedback.

    Joints listed in ``coordinated_indices`` share one scale factor for both
    the target-lead and per-update step limits.  This preserves a multi-joint
    IK direction instead of independently clipping its components and
    distorting the end-effector path.
    """

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
    result = measured + np.clip(
        ramped - measured,
        -max_tracking_error,
        max_tracking_error,
    )
    group = np.asarray(list(coordinated_indices), dtype=np.int64).reshape(-1)
    if group.size == 0:
        return result
    if np.unique(group).size != group.size or np.any(group < 0) or np.any(
        group >= desired.size
    ):
        raise ValueError("coordinated joint indices must be unique and in range")

    desired_lead = desired[group] - measured[group]
    lead_scale = 1.0
    nonzero_lead = np.abs(desired_lead) > 1e-12
    if np.any(nonzero_lead):
        lead_scale = min(
            1.0,
            float(
                np.min(
                    max_tracking_error[group][nonzero_lead]
                    / np.abs(desired_lead[nonzero_lead])
                )
            ),
        )
    tracked = measured[group] + lead_scale * desired_lead
    requested_step = tracked - previous[group]
    step_scale = 1.0
    nonzero_step = np.abs(requested_step) > 1e-12
    if np.any(nonzero_step):
        step_scale = min(
            1.0,
            float(
                np.min(
                    max_step[group][nonzero_step]
                    / np.abs(requested_step[nonzero_step])
                )
            ),
        )
    result[group] = previous[group] + step_scale * requested_step

    # If feedback moved away from the prior target, the step bound alone can
    # leave the new target outside its tracking window.  Pull the whole group
    # back toward feedback with one scale rather than clipping joints alone.
    final_lead = result[group] - measured[group]
    final_scale = 1.0
    nonzero_final = np.abs(final_lead) > 1e-12
    if np.any(nonzero_final):
        final_scale = min(
            1.0,
            float(
                np.min(
                    max_tracking_error[group][nonzero_final]
                    / np.abs(final_lead[nonzero_final])
                )
            ),
        )
    result[group] = measured[group] + final_scale * final_lead
    return result


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


def rate_limited_unit_interval(
    requested: float,
    previous: float,
    rise_rate_s: float,
    fall_rate_s: float,
    dt_s: float,
    deadband: float = 0.0,
) -> float:
    """Move a normalized state without allowing a noisy threshold to chatter."""

    target = float(requested)
    prior = float(previous)
    rise = float(rise_rate_s)
    fall = float(fall_rate_s)
    dt = float(dt_s)
    band = float(deadband)
    if not all(np.isfinite(value) for value in (target, prior, rise, fall, dt, band)):
        raise ValueError("normalized rate-limit inputs must be finite")
    if rise <= 0.0 or fall <= 0.0 or dt <= 0.0 or band < 0.0:
        raise ValueError(
            "normalized rate limits/dt must be positive and deadband non-negative"
        )
    target = float(np.clip(target, 0.0, 1.0))
    prior = float(np.clip(prior, 0.0, 1.0))
    delta = target - prior
    if abs(delta) <= band:
        return prior
    rate = rise if delta > 0.0 else fall
    return float(
        np.clip(prior + np.clip(delta, -rate * dt, rate * dt), 0.0, 1.0)
    )


def gate_outward_scalar_increment(
    requested: float,
    previous: float,
    gate: float,
) -> float:
    """Gate only motion farther from zero; never manufacture a return command."""

    target = float(requested)
    prior = float(previous)
    scale = float(gate)
    if not all(np.isfinite(value) for value in (target, prior, scale)):
        raise ValueError("outward gate inputs must be finite")
    scale = float(np.clip(scale, 0.0, 1.0))
    same_direction = target * prior >= 0.0
    moving_outward = same_direction and abs(target) > abs(prior)
    if not moving_outward:
        return target
    return float(prior + scale * (target - prior))


def task_space_backpressure(
    signed_error: Sequence[float],
    requested_increment: Sequence[float],
    max_tracking_error: Sequence[float],
    start_fraction: float,
) -> float:
    """Reduce only task increments that add to an existing tracking backlog."""

    error = np.asarray(signed_error, dtype=np.float64).reshape(-1)
    increment = np.asarray(requested_increment, dtype=np.float64).reshape(-1)
    limit = np.asarray(max_tracking_error, dtype=np.float64).reshape(-1)
    start = float(start_fraction)
    if not error.shape == increment.shape == limit.shape:
        raise ValueError("task backpressure vectors must have matching shapes")
    if not all(np.all(np.isfinite(value)) for value in (error, increment, limit)):
        raise ValueError("task backpressure vectors must be finite")
    if np.any(limit <= 0.0) or not np.isfinite(start) or not 0.0 <= start < 1.0:
        raise ValueError("task backpressure limits/start fraction are invalid")

    # Returning toward the achieved pose must never be blocked. Only command
    # increments that add to an existing signed error consume headroom.
    increasing = increment * error > 0.0
    if not np.any(increasing):
        return 1.0
    ratio = np.abs(error[increasing]) / limit[increasing]
    normalized = np.clip((ratio - start) / (1.0 - start), 0.0, 1.0)
    factors = 1.0 - np.asarray(
        [minimum_jerk_blend(float(value)) for value in normalized],
        dtype=np.float64,
    )
    return float(np.min(factors)) if factors.size else 1.0


def independent_planar_task_backpressure(
    signed_task_error: Sequence[float],
    requested_planar_increment: Sequence[float],
    vertical_height_profile: Sequence[float],
    max_tracking_error: Sequence[float],
    start_fraction: float,
) -> np.ndarray:
    """Apply anti-windup independently to forward and vertical motion.

    The semantic task contains one forward error followed by one error for
    each protected link height.  A single scalar backpressure factor made a
    lagging height freeze an otherwise safe forward return (and vice versa).
    Keeping two factors preserves the safety constraint without coupling the
    two operator intents.
    """

    error = np.asarray(signed_task_error, dtype=np.float64).reshape(-1)
    increment = np.asarray(
        requested_planar_increment, dtype=np.float64
    ).reshape(-1)
    height_profile = np.asarray(
        vertical_height_profile, dtype=np.float64
    ).reshape(-1)
    limit = np.asarray(max_tracking_error, dtype=np.float64).reshape(-1)
    if increment.size != 2 or error.size != 1 + height_profile.size:
        raise ValueError(
            "planar backpressure expects [forward, lift] and matching heights"
        )
    if limit.shape != error.shape:
        raise ValueError("planar task error and tracking limits must match")
    if not all(
        np.all(np.isfinite(value))
        for value in (error, increment, height_profile, limit)
    ):
        raise ValueError("planar backpressure inputs must be finite")

    forward_factor = task_space_backpressure(
        error[:1],
        increment[:1],
        limit[:1],
        start_fraction,
    )
    height_increment = height_profile * increment[1]
    vertical_factor = task_space_backpressure(
        error[1:],
        height_increment,
        limit[1:],
        start_fraction,
    )
    return np.asarray([forward_factor, vertical_factor], dtype=np.float64)


def wrapped_rate_motion_hold(
    current_delta_rad: float,
    previous_delta_rad: float | None,
    dt_s: float,
    hold_active: bool,
    release_count: int,
    enter_rate_rad_s: float,
    release_rate_rad_s: float,
    release_samples: int,
    observation_valid: bool,
    activation_enabled: bool = True,
) -> tuple[bool, int, float]:
    """Latch a hold while an armed periodic-angle signal moves or is unreliable.

    ``activation_enabled`` only gates a new latch.  An existing hold remains
    active until the normal stable-sample release conditions are satisfied.
    This keeps invalid observations safe during a real wrist gesture without
    allowing small pose-estimator noise to start a long vertical hold.
    """

    values = np.asarray(
        [current_delta_rad, dt_s, enter_rate_rad_s, release_rate_rad_s],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("wrapped-rate hold inputs must be finite")
    if (
        dt_s <= 0.0
        or enter_rate_rad_s <= 0.0
        or release_rate_rad_s < 0.0
        or release_rate_rad_s >= enter_rate_rad_s
        or int(release_samples) < 1
        or int(release_count) < 0
    ):
        raise ValueError("wrapped-rate hold thresholds/counts are invalid")

    if previous_delta_rad is None:
        angular_rate = 0.0
    else:
        previous = float(previous_delta_rad)
        if not np.isfinite(previous):
            raise ValueError("previous wrapped angle must be finite or None")
        angular_rate = wrap_angle_radians(current_delta_rad - previous) / dt_s

    if not hold_active and not activation_enabled:
        return False, 0, float(angular_rate)
    if not observation_valid or abs(angular_rate) >= enter_rate_rad_s:
        return True, 0, float(angular_rate)
    if not hold_active:
        return False, 0, float(angular_rate)
    if abs(angular_rate) <= release_rate_rad_s:
        next_count = int(release_count) + 1
        if next_count >= int(release_samples):
            return False, 0, float(angular_rate)
        return True, next_count, float(angular_rate)
    return True, 0, float(angular_rate)


def rotation_compensated_scalar_observation(
    raw_value: float,
    previous_intent: float,
    observation_bias: float,
    hold_active: bool,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Hold an intent during rotation and rebase its raw observation.

    Updating the bias on every held frame preserves the last accepted command
    while the visual landmark moves for a non-translational reason.  Once the
    hold releases, subsequent operator translation is measured relative to the
    new wrist orientation rather than causing a delayed correction jump.
    """

    values = np.asarray(
        [raw_value, previous_intent, observation_bias, lower_bound, upper_bound],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or lower_bound >= upper_bound:
        raise ValueError("rotation-compensated scalar inputs are invalid")
    previous = float(np.clip(previous_intent, lower_bound, upper_bound))
    if hold_active:
        bias = previous - float(raw_value)
        return previous, float(bias)
    corrected = float(
        np.clip(float(raw_value) + float(observation_bias), lower_bound, upper_bound)
    )
    return corrected, float(observation_bias)


def wrist_height_release_ready(
    wrist_observation_valid: bool,
    hold_active: bool,
    raw_lift_rate_m_s: float,
    maximum_release_lift_rate_m_s: float,
) -> bool:
    """Validate release observations without swallowing vertical gestures.

    Wrist-angle stability already has its own multi-sample confirmation in
    :func:`wrapped_rate_motion_hold`.  Using visual height rate as a second
    release veto made a real downward gesture continuously rebase itself into
    the height bias, so the lift command could remain at its upper bound
    forever.  Height rate remains validated and logged for diagnostics, but it
    must not prolong an already latched wrist hold.
    """

    lift_rate = float(raw_lift_rate_m_s)
    maximum_rate = float(maximum_release_lift_rate_m_s)
    if not np.isfinite(lift_rate) or not np.isfinite(maximum_rate):
        raise ValueError("wrist-height release rates must be finite")
    if maximum_rate <= 0.0:
        raise ValueError("maximum wrist-height release rate must be positive")
    del hold_active, lift_rate, maximum_rate
    return bool(wrist_observation_valid)


def confirmed_stationary_planar_target(
    requested_target_m: Sequence[float],
    current_target_m: Sequence[float],
    current_velocity_m_s: Sequence[float],
    target_deadband_m: Sequence[float],
    velocity_deadband_m_s: Sequence[float],
    previous_count: int,
    confirm_samples: int,
) -> tuple[bool, int]:
    """Latch an operator-idle state after consecutive quiet target samples.

    The resolved-rate controller normally keeps correcting residual task
    error even after the human hand stops.  Because a redundant three-joint
    chain has many nearby solutions, tiny pose-estimator changes can then make
    J2/J3/J4 continuously trade angle.  This detector distinguishes an idle
    operator target from a backpressured or still-moving command so the outer
    task correction can brake to a fixed accepted joint target.
    """

    requested = np.asarray(requested_target_m, dtype=np.float64).reshape(-1)
    current = np.asarray(current_target_m, dtype=np.float64).reshape(-1)
    velocity = np.asarray(current_velocity_m_s, dtype=np.float64).reshape(-1)
    target_deadband = np.asarray(target_deadband_m, dtype=np.float64).reshape(-1)
    velocity_deadband = np.asarray(
        velocity_deadband_m_s, dtype=np.float64
    ).reshape(-1)
    if not (
        requested.shape
        == current.shape
        == velocity.shape
        == target_deadband.shape
        == velocity_deadband.shape
    ):
        raise ValueError("planar idle vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            requested,
            current,
            velocity,
            target_deadband,
            velocity_deadband,
        )
    ):
        raise ValueError("planar idle vectors must be finite")
    if (
        np.any(target_deadband <= 0.0)
        or np.any(velocity_deadband <= 0.0)
        or int(previous_count) < 0
        or int(confirm_samples) < 1
    ):
        raise ValueError("planar idle thresholds/counts are invalid")

    quiet = bool(
        np.all(np.abs(requested - current) <= target_deadband)
        and np.all(np.abs(velocity) <= velocity_deadband)
    )
    next_count = min(
        int(confirm_samples), int(previous_count) + 1
    ) if quiet else 0
    return bool(next_count >= int(confirm_samples)), next_count


def wrist_hold_vertical_planner_state(
    requested_lift_m: float,
    current_command_m: float,
    previous_velocity_m_s: float,
    hold_active: bool,
) -> tuple[float, float]:
    """Freeze both height position and queued velocity during wrist motion."""

    values = np.asarray(
        [requested_lift_m, current_command_m, previous_velocity_m_s],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("vertical planner hold state must be finite")
    if hold_active:
        return float(current_command_m), 0.0
    return float(requested_lift_m), float(previous_velocity_m_s)


def suppress_uncommanded_task_descent(
    joint_velocity: Sequence[float],
    vertical_task_rows: Sequence[Sequence[float]],
    descent_allowed: bool,
    tolerance_m_s: float = 0.0,
) -> np.ndarray:
    """Project joint velocity out of downward task-space half-spaces.

    When the operator is not actively lowering the vertical command, closed-
    loop posture correction may improve one weighted link-height error by
    lowering another link or the TCP.  Iterative orthogonal projection keeps
    every protected height velocity non-negative while retaining as much of
    the requested coordinated J2/J3/J4 motion as possible.
    """

    velocity = np.asarray(joint_velocity, dtype=np.float64).reshape(-1).copy()
    rows = np.asarray(vertical_task_rows, dtype=np.float64)
    tolerance = float(tolerance_m_s)
    if rows.ndim != 2 or rows.shape[1] != velocity.size:
        raise ValueError("vertical task rows must match joint velocity")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(rows)):
        raise ValueError("descent-suppression inputs must be finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("descent-suppression tolerance cannot be negative")
    if descent_allowed or rows.shape[0] == 0:
        return velocity

    for _ in range(max(1, 2 * rows.shape[0])):
        changed = False
        for row in rows:
            norm_squared = float(np.dot(row, row))
            if norm_squared <= 1e-12:
                continue
            task_velocity = float(np.dot(row, velocity))
            if task_velocity < -tolerance:
                velocity += ((-tolerance - task_velocity) / norm_squared) * row
                changed = True
        if not changed:
            break
    if np.any(rows @ velocity < -tolerance - 1e-9):
        # Conflicting or near-singular rows can prevent the small alternating-
        # projection loop from converging numerically.  Holding is safer than
        # leaking a downward command on an unbraked tabletop arm.
        velocity.fill(0.0)
    return velocity


def select_descent_guard_rows(
    vertical_task_rows: Sequence[Sequence[float]],
    frame_ids: Sequence[int],
    end_frame_id: int,
    current_heights_m: Sequence[float],
    table_heights_m: Sequence[float] | None,
    clearance_margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select height constraints without blocking normal planar articulation.

    The TCP is always protected against an uncommanded descent. Intermediate
    links are protected only while they are inside a configurable safety band
    above their validated tabletop heights. Requiring every elevated link to
    have a non-negative instantaneous Z velocity over-constrains J2/J3/J4 and
    can make forward/backward Cartesian motion mathematically impossible.
    The independent full-path table projection remains the hard floor.
    """

    rows = np.asarray(vertical_task_rows, dtype=np.float64)
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    current = np.asarray(current_heights_m, dtype=np.float64).reshape(-1)
    margin = float(clearance_margin_m)
    if rows.ndim != 2 or rows.shape[0] != ids.size or ids.size != current.size:
        raise ValueError("descent-guard rows, frames and heights must align")
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(current)):
        raise ValueError("descent-guard rows and heights must be finite")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("descent-guard clearance margin cannot be negative")

    mask = ids == int(end_frame_id)
    if table_heights_m is not None:
        table = np.asarray(table_heights_m, dtype=np.float64).reshape(-1)
        if table.size != ids.size or not np.all(np.isfinite(table)):
            raise ValueError("descent-guard table heights must align and be finite")
        mask |= (current - table) <= margin
    if not np.any(mask):
        raise ValueError("descent guard is missing the end-effector frame")
    return rows[mask].copy(), mask


def coordinated_acceleration_limited_velocity(
    requested_velocity: Sequence[float],
    previous_velocity: Sequence[float],
    max_velocity: Sequence[float],
    max_acceleration: Sequence[float],
    dt_s: float,
) -> np.ndarray:
    """Limit a coordinated joint velocity without distorting its direction."""

    requested = np.asarray(requested_velocity, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_velocity, dtype=np.float64).reshape(-1)
    velocity_limit = np.asarray(max_velocity, dtype=np.float64).reshape(-1)
    acceleration_limit = np.asarray(max_acceleration, dtype=np.float64).reshape(-1)
    dt = float(dt_s)
    if not (
        requested.shape
        == previous.shape
        == velocity_limit.shape
        == acceleration_limit.shape
    ):
        raise ValueError("coordinated velocity vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (requested, previous, velocity_limit, acceleration_limit)
    ):
        raise ValueError("coordinated velocity vectors must be finite")
    if (
        np.any(velocity_limit <= 0.0)
        or np.any(acceleration_limit <= 0.0)
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("coordinated velocity limits and dt must be positive")

    velocity_scale = 1.0
    active = np.abs(requested) > 1e-12
    if np.any(active):
        velocity_scale = min(
            1.0,
            float(np.min(velocity_limit[active] / np.abs(requested[active]))),
        )
    bounded_requested = velocity_scale * requested
    delta = bounded_requested - previous
    acceleration_scale = 1.0
    changing = np.abs(delta) > 1e-12
    if np.any(changing):
        acceleration_scale = min(
            1.0,
            float(
                np.min(
                    acceleration_limit[changing] * dt / np.abs(delta[changing])
                )
            ),
        )
    result = previous + acceleration_scale * delta
    final_scale = 1.0
    moving = np.abs(result) > 1e-12
    if np.any(moving):
        final_scale = min(
            1.0,
            float(np.min(velocity_limit[moving] / np.abs(result[moving]))),
        )
    return final_scale * result


def acceleration_limited_vector_target(
    requested: Sequence[float],
    previous: Sequence[float],
    previous_velocity: Sequence[float],
    max_velocity: Sequence[float],
    max_acceleration: Sequence[float],
    deadband: Sequence[float],
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the scalar acceleration limiter independently to each task axis."""

    vectors = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            requested,
            previous,
            previous_velocity,
            max_velocity,
            max_acceleration,
            deadband,
        )
    ]
    if not all(value.shape == vectors[0].shape for value in vectors[1:]):
        raise ValueError("acceleration-limit vectors must have matching shapes")
    target = np.empty_like(vectors[0])
    velocity = np.empty_like(vectors[0])
    for index in range(vectors[0].size):
        target[index], velocity[index] = acceleration_limited_scalar_target(
            float(vectors[0][index]),
            float(vectors[1][index]),
            float(vectors[2][index]),
            float(vectors[3][index]),
            float(vectors[4][index]),
            float(vectors[5][index]),
            dt_s,
        )
    return target, velocity


def confirm_vector_reversals(
    requested: Sequence[float],
    accepted: Sequence[float],
    active_direction: Sequence[int],
    candidate_direction: Sequence[int],
    candidate_count: Sequence[int],
    reverse_deadband: Sequence[float],
    confirm_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reject small or short-lived task-direction reversals.

    The absolute palm displacement is a noisy vision signal.  Once an axis has
    started moving in one direction, an opposite target must remain beyond the
    reversal deadband for ``confirm_samples`` consecutive control updates.  A
    same-direction target is accepted immediately, preserving normal response.
    """

    target = np.asarray(requested, dtype=np.float64).reshape(-1)
    stable = np.asarray(accepted, dtype=np.float64).reshape(-1).copy()
    active = np.asarray(active_direction, dtype=np.int8).reshape(-1).copy()
    candidate = np.asarray(candidate_direction, dtype=np.int8).reshape(-1).copy()
    count = np.asarray(candidate_count, dtype=np.int64).reshape(-1).copy()
    deadband = np.asarray(reverse_deadband, dtype=np.float64).reshape(-1)
    if not (
        target.shape
        == stable.shape
        == active.shape
        == candidate.shape
        == count.shape
        == deadband.shape
    ):
        raise ValueError("direction-confirmation vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value)) for value in (target, stable, deadband)
    ):
        raise ValueError("direction-confirmation values must be finite")
    if np.any(deadband < 0.0) or int(confirm_samples) < 1:
        raise ValueError(
            "reversal deadbands must be non-negative and confirmation positive"
        )

    for index in range(target.size):
        delta = float(target[index] - stable[index])
        if abs(delta) <= 1e-12:
            candidate[index] = 0
            count[index] = 0
            continue
        direction = 1 if delta > 0.0 else -1
        if active[index] == 0 or direction == active[index]:
            # The reversal deadband is not a general motion deadband.  Once an
            # axis is moving, every same-direction update must pass through or
            # a slowly moving hand becomes a sequence of deadband-sized steps.
            stable[index] = target[index]
            active[index] = direction
            candidate[index] = 0
            count[index] = 0
            continue
        if abs(delta) <= deadband[index]:
            candidate[index] = 0
            count[index] = 0
            continue
        if candidate[index] == direction:
            count[index] += 1
        else:
            candidate[index] = direction
            count[index] = 1
        if count[index] >= int(confirm_samples):
            stable[index] = target[index]
            active[index] = direction
            candidate[index] = 0
            count[index] = 0
    return stable, active, candidate, count


def acceleration_limited_scalar_target(
    requested: float,
    previous: float,
    previous_velocity: float,
    max_velocity: float,
    max_acceleration: float,
    deadband: float,
    dt_s: float,
) -> tuple[float, float]:
    """Return a velocity/acceleration-limited scalar target.

    The deadband is an arrival tolerance, not a hold zone.  Snapping the outer
    target to the requested value on arrival prevents small errors from
    accumulating until they are released as a visible step.  The 500 Hz joint
    stream interpolator below still makes the motor setpoint continuous.
    """

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
        return float(requested), 0.0
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


def acceleration_limited_scalar_target_with_tracking_window(
    requested: float,
    measured: float,
    previous: float,
    previous_velocity: float,
    max_velocity: float,
    max_acceleration: float,
    deadband: float,
    max_tracking_error: float,
    dt_s: float,
) -> tuple[float, float]:
    """Shape a direct-joint request once, then enforce feedback lead bounds.

    This ordering matters for J1/J6: applying a generic per-frame step before
    the acceleration limiter makes each tiny step look already reachable and
    repeatedly resets its velocity state.  The returned velocity describes
    the accepted target step and is therefore also suitable for stream
    feed-forward.
    """

    values = np.asarray(
        [
            requested,
            measured,
            previous,
            previous_velocity,
            max_velocity,
            max_acceleration,
            deadband,
            max_tracking_error,
            dt_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or max_tracking_error <= 0.0:
        raise ValueError("direct-joint target/tracking inputs are invalid")
    target, _planned_velocity = acceleration_limited_scalar_target(
        requested,
        previous,
        previous_velocity,
        max_velocity,
        max_acceleration,
        deadband,
        dt_s,
    )
    target = float(
        np.clip(
            target,
            measured - max_tracking_error,
            measured + max_tracking_error,
        )
    )
    accepted_velocity = float(
        np.clip(
            (target - previous) / dt_s,
            -max_velocity,
            max_velocity,
        )
    )
    return target, accepted_velocity


def coupled_stream_velocity_limits(
    configured_limit: Sequence[float],
    semantic_max_step: Sequence[float],
    command_interval_s: float,
    outer_velocity_limit: Sequence[float],
    follow_scale: float | Sequence[float],
) -> np.ndarray:
    """Match the high-rate stream to the semantic goal's achievable velocity.

    A stream that is several times faster than the outer goal reaches every
    20--25 Hz sample early and brakes to rest, recreating visible stop-and-go
    motion.  The configured stream values remain hard caps; the returned
    limits add only a small following margin over the actual outer rate.
    """

    configured = np.asarray(configured_limit, dtype=np.float64).reshape(-1)
    max_step = np.asarray(semantic_max_step, dtype=np.float64).reshape(-1)
    outer_limit = np.asarray(outer_velocity_limit, dtype=np.float64).reshape(-1)
    if not configured.shape == max_step.shape == outer_limit.shape:
        raise ValueError("coupled stream-limit vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value)) for value in (configured, max_step, outer_limit)
    ):
        raise ValueError("coupled stream-limit vectors must be finite")
    interval = float(command_interval_s)
    scale = np.asarray(follow_scale, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(configured.shape, float(scale), dtype=np.float64)
    else:
        scale = scale.reshape(-1)
    if (
        np.any(configured <= 0.0)
        or np.any(max_step <= 0.0)
        or np.any(outer_limit <= 0.0)
        or scale.shape != configured.shape
        or not np.all(np.isfinite(scale))
        or not np.isfinite(interval)
        or interval <= 0.0
        or np.any(scale < 1.0)
    ):
        raise ValueError("coupled stream limits, interval and follow scale are invalid")
    semantic_rate = np.minimum(max_step / interval, outer_limit)
    return np.minimum(configured, scale * semantic_rate)


def adaptive_goal_velocity_hold_s(
    command_interval_s: float,
    observed_update_period_s: float,
    stale_cycles: float,
    maximum_hold_s: float,
) -> float:
    """Keep feed-forward alive across the measured vision-frame interval."""

    interval = float(command_interval_s)
    observed = float(observed_update_period_s)
    cycles = float(stale_cycles)
    maximum = float(maximum_hold_s)
    if not all(np.isfinite(value) for value in (interval, observed, cycles, maximum)):
        raise ValueError("goal-velocity hold settings must be finite")
    if interval <= 0.0 or observed <= 0.0 or cycles < 1.0 or maximum <= 0.0:
        raise ValueError("goal-velocity hold settings are invalid")
    nominal_hold = cycles * interval
    measured_hold = 1.5 * observed
    return float(min(maximum, max(nominal_hold, measured_hold)))


def filtered_goal_velocity(
    requested_position: Sequence[float],
    previous_position: Sequence[float],
    previous_velocity: Sequence[float],
    max_velocity: Sequence[float],
    filter_alpha: float,
    dt_s: float,
    coordinated_indices: Sequence[int] = (),
) -> np.ndarray:
    """Estimate a bounded joint-goal velocity from consecutive outer targets."""

    requested = np.asarray(requested_position, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_position, dtype=np.float64).reshape(-1)
    prior_velocity = np.asarray(previous_velocity, dtype=np.float64).reshape(-1)
    velocity_limit = np.asarray(max_velocity, dtype=np.float64).reshape(-1)
    if not requested.shape == previous.shape == prior_velocity.shape == velocity_limit.shape:
        raise ValueError("goal-velocity vectors must have matching shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (requested, previous, prior_velocity, velocity_limit)
    ):
        raise ValueError("goal-velocity vectors must be finite")
    alpha = float(filter_alpha)
    dt = float(dt_s)
    if (
        np.any(velocity_limit <= 0.0)
        or not np.isfinite(alpha)
        or not 0.0 < alpha <= 1.0
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("goal-velocity limits, alpha and dt are invalid")

    raw_velocity = (requested - previous) / dt
    limited_velocity = np.clip(raw_velocity, -velocity_limit, velocity_limit)
    group = np.asarray(list(coordinated_indices), dtype=np.int64).reshape(-1)
    if group.size:
        if np.unique(group).size != group.size or np.any(group < 0) or np.any(
            group >= requested.size
        ):
            raise ValueError("coordinated goal-velocity indices are invalid")
        group_raw = raw_velocity[group]
        nonzero = np.abs(group_raw) > 1e-12
        group_scale = 1.0
        if np.any(nonzero):
            group_scale = min(
                1.0,
                float(
                    np.min(
                        velocity_limit[group][nonzero]
                        / np.abs(group_raw[nonzero])
                    )
                ),
            )
        limited_velocity[group] = group_scale * group_raw

    # Do not carry queued velocity through a genuine target reversal.  For the
    # planar group, a negative dot product means the joint-space path reversed.
    reset = raw_velocity * prior_velocity < 0.0
    if group.size and float(np.dot(raw_velocity[group], prior_velocity[group])) < 0.0:
        reset[group] = True
    filtered_from = prior_velocity.copy()
    filtered_from[reset] = 0.0
    result = alpha * limited_velocity + (1.0 - alpha) * filtered_from
    return np.clip(result, -velocity_limit, velocity_limit)


def smoothed_semantic_goal_velocity(
    requested_position: Sequence[float],
    previous_position: Sequence[float],
    previous_velocity: Sequence[float],
    max_velocity: Sequence[float],
    filter_alpha: float,
    dt_s: float,
    j1_velocity_rad_s: float,
    j6_velocity_rad_s: float,
) -> np.ndarray:
    """Build feed-forward velocity without bypassing planar interpolation.

    J1 and J6 already have dedicated scalar acceleration limiters, so their
    accepted velocities pass through unchanged.  J2/J3/J4 are filtered as one
    coordinated vector.  This is important when inference supplies only about
    five target updates per second: copying the raw resolved-rate vector into
    the 500 Hz stream turns every camera frame into a visible velocity step.
    """

    result = filtered_goal_velocity(
        requested_position,
        previous_position,
        previous_velocity,
        max_velocity,
        filter_alpha,
        dt_s,
        coordinated_indices=(1, 2, 3),
    )
    direct = np.asarray([j1_velocity_rad_s, j6_velocity_rad_s], dtype=np.float64)
    if not np.all(np.isfinite(direct)):
        raise ValueError("direct semantic goal velocities must be finite")
    result[0] = float(direct[0])
    result[5] = float(direct[1])
    return result


def jerk_limited_joint_target(
    requested: Sequence[float],
    previous: Sequence[float],
    previous_velocity: Sequence[float],
    previous_acceleration: Sequence[float],
    max_velocity: Sequence[float],
    max_acceleration: Sequence[float],
    max_jerk: Sequence[float],
    dt_s: float,
    requested_velocity: Sequence[float] | None = None,
    position_correction_gain_s: float | None = None,
    position_correction_max_velocity: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance a continuous joint setpoint with bounded position/velocity tracking."""

    vectors = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            requested,
            previous,
            previous_velocity,
            previous_acceleration,
            max_velocity,
            max_acceleration,
            max_jerk,
        )
    ]
    if not all(value.shape == vectors[0].shape for value in vectors[1:]):
        raise ValueError("joint-stream vectors must have matching shapes")
    if not all(np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("joint-stream vectors must contain finite values")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("joint-stream dt must be positive and finite")
    if any(np.any(value <= 0.0) for value in vectors[4:]):
        raise ValueError("joint-stream velocity, acceleration and jerk limits must be positive")

    goal, position, velocity, acceleration, velocity_limit, acceleration_limit, jerk_limit = vectors
    goal_velocity = (
        np.zeros_like(goal)
        if requested_velocity is None
        else np.asarray(requested_velocity, dtype=np.float64).reshape(-1)
    )
    if goal_velocity.shape != goal.shape or not np.all(np.isfinite(goal_velocity)):
        raise ValueError("requested joint-stream velocity must match the goal")
    goal_velocity = np.clip(goal_velocity, -velocity_limit, velocity_limit)
    error = goal - position
    # Feed-forward preserves motion across consecutive outer-loop samples.
    # Legacy callers retain stopping-distance feedback; the semantic path uses
    # a much smaller proportional correction so it cannot race to each frame.
    if position_correction_gain_s is None:
        correction_speed = np.minimum(
            velocity_limit,
            np.sqrt(2.0 * acceleration_limit * np.abs(error)),
        )
    else:
        correction_gain = float(position_correction_gain_s)
        correction_limit = (
            velocity_limit
            if position_correction_max_velocity is None
            else np.asarray(
                position_correction_max_velocity, dtype=np.float64
            ).reshape(-1)
        )
        if (
            not np.isfinite(correction_gain)
            or correction_gain <= 0.0
            or correction_limit.shape != goal.shape
            or not np.all(np.isfinite(correction_limit))
            or np.any(correction_limit <= 0.0)
        ):
            raise ValueError("joint-stream position correction settings are invalid")
        correction_speed = np.minimum(
            np.minimum(correction_limit, velocity_limit),
            correction_gain * np.abs(error),
        )
    desired_velocity = np.clip(
        goal_velocity + np.sign(error) * correction_speed,
        -velocity_limit,
        velocity_limit,
    )
    desired_acceleration = np.clip(
        (desired_velocity - velocity) / dt,
        -acceleration_limit,
        acceleration_limit,
    )
    acceleration_step = np.clip(
        desired_acceleration - acceleration,
        -jerk_limit * dt,
        jerk_limit * dt,
    )
    next_acceleration = np.clip(
        acceleration + acceleration_step,
        -acceleration_limit,
        acceleration_limit,
    )
    next_velocity = np.clip(
        velocity + next_acceleration * dt,
        -velocity_limit,
        velocity_limit,
    )
    step = 0.5 * (velocity + next_velocity) * dt
    next_position = position + step

    # A setpoint generator has no physical inertia to honour.  During a goal
    # reversal, do not preserve stale internal momentum: it otherwise produces
    # several held samples followed by a visible catch-up step, and its logged
    # velocity points opposite to the current operator intent.  Position stays
    # continuous while the next cycle accelerates toward the new goal.
    moving_away = step * error < 0.0
    next_position[moving_away] = position[moving_away]
    next_velocity[moving_away] = 0.0
    next_acceleration[moving_away] = 0.0
    at_goal = np.abs(error) <= 1e-10
    next_position[at_goal] = goal[at_goal]
    arrived = np.logical_and(step * error > 0.0, np.abs(step) >= np.abs(error))
    next_position[arrived] = goal[arrived]
    stationary = np.logical_and.reduce(
        (
            at_goal,
            np.abs(goal_velocity) <= 1e-8,
            np.abs(velocity) <= 1e-8,
            np.abs(acceleration) <= 1e-6,
        )
    )
    next_position[stationary] = goal[stationary]
    next_velocity[stationary] = 0.0
    next_acceleration[stationary] = 0.0
    return next_position, next_velocity, next_acceleration


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


def validate_wrapped_angle_observation(
    value: float,
    previous_value: float | None,
    dt_s: float,
    max_rate_rad_s: float,
    max_jump_rad: float,
) -> tuple[bool, float]:
    """Reject implausible periodic-angle observations instead of integrating them.

    ``filtered_wrapped_angle_delta`` intentionally limits innovations, but a
    run of bad depth frames can still accumulate into a plausible-looking
    command.  This first-stage gate rejects the complete frame when its raw
    wrapped innovation exceeds either a physical rate limit or an absolute
    per-frame jump limit.  The returned innovation is useful for diagnostics.
    """

    values = np.asarray(
        [value, dt_s, max_rate_rad_s, max_jump_rad], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("wrist observation gate inputs must be finite")
    if dt_s <= 0.0 or max_rate_rad_s <= 0.0 or not 0.0 < max_jump_rad <= np.pi:
        raise ValueError("wrist observation rate/jump limits must be positive")
    if previous_value is None:
        return True, 0.0
    previous = float(previous_value)
    if not np.isfinite(previous):
        raise ValueError("previous wrist observation must be finite or None")
    innovation = wrap_angle_radians(float(value) - previous)
    allowed = min(float(max_jump_rad), float(max_rate_rad_s) * float(dt_s))
    return bool(abs(innovation) <= allowed), float(innovation)


def confirm_directional_wrist_twist(
    filtered_delta_rad: float,
    active_direction: int,
    held_delta_rad: float,
    candidate_direction: int,
    candidate_count: int,
    activate_rad: float,
    release_rad: float,
    confirm_samples: int,
    reverse_confirm_samples: int | None = None,
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
    reverse_samples = (
        int(confirm_samples)
        if reverse_confirm_samples is None
        else int(reverse_confirm_samples)
    )
    if confirm_samples < 1 or reverse_samples < 1 or candidate_count < 0:
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
    required_samples = (
        reverse_samples
        if active_direction != 0
        and candidate_direction != 0
        and candidate_direction != active_direction
        else int(confirm_samples)
    )
    if candidate_count < required_samples:
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
        # Conservative intersection target from Seeed's published B601-DM
        # range table: J1 +/-149 deg, J2/J3 -206..0 deg, J4/J5/J6 +/-85.9 deg.
        # The current official DM URDF is intersected with this envelope after
        # loading; joint_tracking_bounds then applies the additional margin.
        self.configured_safe_joint_lower = _vector(
            node.declare_parameter(
                "b601_safe_joint_lower_rad",
                [-2.6005, -3.5954, -3.5954, -1.4992, -1.4992, -1.4992],
            ).value,
            self.ARM_DOF,
            "b601_safe_joint_lower_rad",
        )
        self.configured_safe_joint_upper = _vector(
            node.declare_parameter(
                "b601_safe_joint_upper_rad",
                [2.6005, 0.0, 0.0, 1.4992, 1.4992, 1.4992],
            ).value,
            self.ARM_DOF,
            "b601_safe_joint_upper_rad",
        )
        if np.any(
            self.configured_safe_joint_lower >= self.configured_safe_joint_upper
        ):
            raise ValueError(
                "b601_safe_joint_lower_rad must be below "
                "b601_safe_joint_upper_rad for every joint"
            )
        self.control_rate_hz = float(
            node.declare_parameter("b601_control_rate_hz", 500.0).value
        )
        if not np.isfinite(self.control_rate_hz) or self.control_rate_hz <= 0.0:
            raise ValueError("b601_control_rate_hz must be positive and finite")
        # These high-rate limits shape the setpoint sent on every MotorBridge
        # cycle.  Semantic J1/J4/J6 limits below remain authoritative when they
        # are lower, preserving existing launch-command tuning.
        self.stream_max_velocity = _positive_vector(
            node.declare_parameter(
                "b601_stream_max_velocity_rad_s",
                [0.25, 0.30, 0.35, 0.15, 0.10, 0.25],
            ).value,
            self.ARM_DOF,
            "b601_stream_max_velocity_rad_s",
        )
        self.stream_max_acceleration = _positive_vector(
            node.declare_parameter(
                "b601_stream_max_acceleration_rad_s2",
                [2.0, 2.5, 3.0, 1.2, 0.8, 2.0],
            ).value,
            self.ARM_DOF,
            "b601_stream_max_acceleration_rad_s2",
        )
        self.stream_max_jerk = _positive_vector(
            node.declare_parameter(
                "b601_stream_max_jerk_rad_s3",
                [80.0, 100.0, 120.0, 50.0, 30.0, 80.0],
            ).value,
            self.ARM_DOF,
            "b601_stream_max_jerk_rad_s3",
        )
        self.stream_outer_velocity_scale = float(
            node.declare_parameter("b601_stream_outer_velocity_scale", 1.35).value
        )
        self.stream_planar_outer_velocity_scale = float(
            node.declare_parameter(
                "b601_stream_planar_outer_velocity_scale", 1.10
            ).value
        )
        self.stream_goal_velocity_filter_alpha = float(
            node.declare_parameter(
                "b601_stream_goal_velocity_filter_alpha", 0.45
            ).value
        )
        self.stream_goal_velocity_stale_cycles = float(
            node.declare_parameter(
                "b601_stream_goal_velocity_stale_cycles", 2.0
            ).value
        )
        self.stream_goal_period_filter_alpha = float(
            node.declare_parameter(
                "b601_stream_goal_period_filter_alpha", 0.20
            ).value
        )
        self.stream_goal_velocity_max_hold_s = float(
            node.declare_parameter(
                "b601_stream_goal_velocity_max_hold_s", 0.20
            ).value
        )
        self.stream_position_correction_gain_s = float(
            node.declare_parameter(
                "b601_stream_position_correction_gain_s", 5.0
            ).value
        )
        self.stream_position_correction_max_fraction = float(
            node.declare_parameter(
                "b601_stream_position_correction_max_fraction", 0.35
            ).value
        )
        self.stream_planar_position_correction_max_fraction = float(
            node.declare_parameter(
                "b601_stream_planar_position_correction_max_fraction", 0.20
            ).value
        )
        if (
            not np.isfinite(self.stream_outer_velocity_scale)
            or self.stream_outer_velocity_scale < 1.0
            or not np.isfinite(self.stream_planar_outer_velocity_scale)
            or self.stream_planar_outer_velocity_scale < 1.0
            or not np.isfinite(self.stream_goal_velocity_filter_alpha)
            or not 0.0 < self.stream_goal_velocity_filter_alpha <= 1.0
            or not np.isfinite(self.stream_goal_velocity_stale_cycles)
            or self.stream_goal_velocity_stale_cycles < 1.0
            or not np.isfinite(self.stream_goal_period_filter_alpha)
            or not 0.0 < self.stream_goal_period_filter_alpha <= 1.0
            or not np.isfinite(self.stream_goal_velocity_max_hold_s)
            or self.stream_goal_velocity_max_hold_s <= 0.0
            or not np.isfinite(self.stream_position_correction_gain_s)
            or self.stream_position_correction_gain_s <= 0.0
            or not np.isfinite(self.stream_position_correction_max_fraction)
            or not 0.0 < self.stream_position_correction_max_fraction <= 1.0
            or not np.isfinite(
                self.stream_planar_position_correction_max_fraction
            )
            or not 0.0
            < self.stream_planar_position_correction_max_fraction
            <= self.stream_position_correction_max_fraction
        ):
            raise ValueError(
                "B601 stream follow scale/stale cycles must be >= 1 and "
                "goal-velocity alpha/correction fractions must be valid; the "
                "planar correction fraction cannot exceed the global fraction"
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
        if (
            not np.isfinite(self.command_interval_s)
            or self.command_interval_s <= 0.0
            or self.stream_goal_velocity_max_hold_s < self.command_interval_s
        ):
            raise ValueError(
                "arm_command_interval_s must be positive and cannot exceed "
                "b601_stream_goal_velocity_max_hold_s"
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
        self.palm_filter_fast_alpha = float(
            node.declare_parameter("palm_filter_fast_alpha", 0.72).value
        )
        self.palm_filter_speed_scale_m_s = float(
            node.declare_parameter("palm_filter_speed_scale_m_s", 0.30).value
        )
        self.palm_filter_max_innovation_m = _positive_vector(
            node.declare_parameter(
                "palm_filter_max_innovation_m", [0.030, 0.030, 0.030]
            ).value,
            3,
            "palm_filter_max_innovation_m",
        )
        if (
            not np.isfinite(self.palm_filter_alpha)
            or not 0.0 < self.palm_filter_alpha <= 1.0
            or not np.isfinite(self.palm_filter_fast_alpha)
            or not self.palm_filter_alpha <= self.palm_filter_fast_alpha <= 1.0
            or not np.isfinite(self.palm_filter_speed_scale_m_s)
            or self.palm_filter_speed_scale_m_s <= 0.0
        ):
            raise ValueError(
                "palm filter alpha must be in (0, 1], fast alpha cannot be "
                "smaller, and speed scale must be positive"
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
                    "b601_semantic_linear_deadband_m", [0.012, 0.008, 0.008]
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
                "b601_wrist_twist_filter_alpha", 0.18
            ).value
        )
        self.wrist_twist_max_innovation_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_max_innovation_deg", 12.0
                ).value
            )
        )
        self.wrist_twist_max_raw_rate_rad_s = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_max_raw_rate_deg_s", 180.0
                ).value
            )
        )
        self.wrist_twist_max_raw_jump_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_wrist_twist_max_raw_jump_deg", 25.0
                ).value
            )
        )
        self.wrist_twist_translation_gate_m_s = float(
            node.declare_parameter(
                "b601_wrist_twist_translation_gate_m_s", 0.050
            ).value
        )
        self.wrist_twist_stationary_confirm_samples = int(
            node.declare_parameter(
                "b601_wrist_twist_stationary_confirm_samples", 3
            ).value
        )
        self.vertical_wrist_decoupling_enabled = bool(
            node.declare_parameter(
                "b601_vertical_wrist_decoupling_enabled", True
            ).value
        )
        self.vertical_wrist_hold_rate_rad_s = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_vertical_wrist_hold_rate_deg_s", 3.0
                ).value
            )
        )
        self.vertical_wrist_release_rate_rad_s = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_vertical_wrist_release_rate_deg_s", 1.0
                ).value
            )
        )
        self.vertical_wrist_release_samples = int(
            node.declare_parameter(
                "b601_vertical_wrist_release_samples", 5
            ).value
        )
        self.vertical_wrist_release_lift_rate_m_s = float(
            node.declare_parameter(
                "b601_vertical_wrist_release_lift_rate_m_s", 0.012
            ).value
        )
        self.vertical_wrist_hold_min_delta_rad = np.deg2rad(
            float(
                node.declare_parameter(
                    "b601_vertical_wrist_hold_min_delta_deg", 6.0
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
                "b601_wrist_twist_confirm_samples", 3
            ).value
        )
        self.wrist_twist_reverse_confirm_samples = int(
            node.declare_parameter(
                "b601_wrist_twist_reverse_confirm_samples", 6
            ).value
        )
        self.wrist_twist_soft_start_s = float(
            node.declare_parameter("b601_wrist_twist_soft_start_s", 0.30).value
        )
        if (
            not np.isfinite(self.wrist_twist_filter_alpha)
            or not 0.0 < self.wrist_twist_filter_alpha <= 1.0
            or not np.isfinite(self.wrist_twist_max_innovation_rad)
            or not 0.0 < self.wrist_twist_max_innovation_rad <= np.pi
            or not np.isfinite(self.wrist_twist_max_raw_rate_rad_s)
            or self.wrist_twist_max_raw_rate_rad_s <= 0.0
            or not np.isfinite(self.wrist_twist_max_raw_jump_rad)
            or not 0.0 < self.wrist_twist_max_raw_jump_rad <= np.pi
            or not np.isfinite(self.wrist_twist_translation_gate_m_s)
            or self.wrist_twist_translation_gate_m_s <= 0.0
            or self.wrist_twist_stationary_confirm_samples < 1
            or not np.isfinite(self.vertical_wrist_hold_rate_rad_s)
            or self.vertical_wrist_hold_rate_rad_s <= 0.0
            or not np.isfinite(self.vertical_wrist_release_rate_rad_s)
            or self.vertical_wrist_release_rate_rad_s < 0.0
            or self.vertical_wrist_release_rate_rad_s
            >= self.vertical_wrist_hold_rate_rad_s
            or self.vertical_wrist_release_samples < 1
            or not np.isfinite(self.vertical_wrist_release_lift_rate_m_s)
            or self.vertical_wrist_release_lift_rate_m_s <= 0.0
            or not np.isfinite(self.vertical_wrist_hold_min_delta_rad)
            or self.vertical_wrist_hold_min_delta_rad < 0.0
            or self.vertical_wrist_hold_min_delta_rad
            > self.wrist_twist_activate_rad
            or not np.isfinite(self.wrist_twist_activate_rad)
            or not np.isfinite(self.wrist_twist_release_rad)
            or self.wrist_twist_release_rad < 0.0
            or self.wrist_twist_activate_rad <= self.wrist_twist_release_rad
            or self.wrist_twist_confirm_samples < 1
            or self.wrist_twist_reverse_confirm_samples
            < self.wrist_twist_confirm_samples
            or not np.isfinite(self.wrist_twist_soft_start_s)
            or self.wrist_twist_soft_start_s <= 0.0
        ):
            raise ValueError(
                "B601 wrist-twist filter alpha must be in (0, 1] and "
                "angle limits/rates and translation gate must be positive; "
                "vertical wrist release rate must be below its hold rate and "
                "the release lift-rate limit must be positive; the vertical "
                "hold arming angle must be between zero and activation; "
                "activation must exceed release; reversal confirmation must "
                "be at least the normal positive confirmation count"
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
                "b601_semantic_planar_input_scale", [0.45, 1.0]
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
                "b601_semantic_planar_command_max_velocity_m_s", [0.060, 0.035]
            ).value,
            2,
            "b601_semantic_planar_command_max_velocity_m_s",
        )
        self.semantic_planar_command_max_acceleration = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_command_max_acceleration_m_s2",
                [0.32, 0.16],
            ).value,
            2,
            "b601_semantic_planar_command_max_acceleration_m_s2",
        )
        self.semantic_planar_command_deadband = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_command_deadband_m", [0.0015, 0.0015]
            ).value,
            2,
            "b601_semantic_planar_command_deadband_m",
        )
        self.semantic_planar_idle_target_deadband = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_idle_target_deadband_m",
                [0.0025, 0.0030],
            ).value,
            2,
            "b601_semantic_planar_idle_target_deadband_m",
        )
        self.semantic_planar_idle_velocity_deadband = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_idle_velocity_deadband_m_s",
                [0.0030, 0.0030],
            ).value,
            2,
            "b601_semantic_planar_idle_velocity_deadband_m_s",
        )
        self.semantic_planar_idle_confirm_samples = int(
            node.declare_parameter(
                "b601_semantic_planar_idle_confirm_samples", 3
            ).value
        )
        if self.semantic_planar_idle_confirm_samples < 1:
            raise ValueError(
                "b601_semantic_planar_idle_confirm_samples must be positive"
            )
        self.semantic_planar_tracking_error = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_tracking_error_m", [0.025, 0.025]
            ).value,
            2,
            "b601_semantic_planar_tracking_error_m",
        )
        self.semantic_planar_backpressure_start_fraction = float(
            node.declare_parameter(
                "b601_semantic_planar_backpressure_start_fraction", 0.55
            ).value
        )
        self.semantic_planar_backpressure_rates = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_backpressure_rates_s", [2.0, 4.0]
            ).value,
            2,
            "b601_semantic_planar_backpressure_rates_s",
        )
        self.semantic_clearance_gate_rates = _positive_vector(
            node.declare_parameter(
                "b601_semantic_clearance_gate_rates_s", [1.5, 3.0]
            ).value,
            2,
            "b601_semantic_clearance_gate_rates_s",
        )
        self.semantic_clearance_gate_deadband = float(
            node.declare_parameter(
                "b601_semantic_clearance_gate_deadband", 0.03
            ).value
        )
        self.semantic_planar_task_error_gain = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_task_error_gain_s", [2.0, 2.0]
            ).value,
            2,
            "b601_semantic_planar_task_error_gain_s",
        )
        self.semantic_planar_task_correction_max_velocity = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_task_correction_max_velocity_m_s",
                [0.025, 0.020],
            ).value,
            2,
            "b601_semantic_planar_task_correction_max_velocity_m_s",
        )
        self.semantic_vertical_descent_guard_enabled = bool(
            node.declare_parameter(
                "b601_semantic_vertical_descent_guard_enabled", True
            ).value
        )
        self.semantic_vertical_descent_intent_deadband_m_s = float(
            node.declare_parameter(
                "b601_semantic_vertical_descent_intent_deadband_m_s", 0.0015
            ).value
        )
        self.semantic_vertical_descent_guard_clearance_margin_m = float(
            node.declare_parameter(
                "b601_semantic_vertical_descent_guard_clearance_margin_m",
                0.010,
            ).value
        )
        if (
            not np.isfinite(
                self.semantic_vertical_descent_intent_deadband_m_s
            )
            or self.semantic_vertical_descent_intent_deadband_m_s < 0.0
            or not np.isfinite(
                self.semantic_vertical_descent_guard_clearance_margin_m
            )
            or self.semantic_vertical_descent_guard_clearance_margin_m < 0.0
        ):
            raise ValueError(
                "B601 vertical descent intent deadband and clearance margin "
                "must be finite and non-negative"
            )
        self.semantic_planar_joint_max_acceleration = _positive_vector(
            node.declare_parameter(
                "b601_semantic_planar_joint_max_acceleration_rad_s2",
                [0.45, 0.60, 0.20],
            ).value,
            3,
            "b601_semantic_planar_joint_max_acceleration_rad_s2",
        )
        self.semantic_planar_follower_min_mobility_scale = float(
            node.declare_parameter(
                "b601_semantic_planar_follower_min_mobility_scale", 0.30
            ).value
        )
        if (
            not np.isfinite(self.semantic_planar_backpressure_start_fraction)
            or not 0.0
            <= self.semantic_planar_backpressure_start_fraction
            < 1.0
            or not np.isfinite(self.semantic_clearance_gate_deadband)
            or not 0.0 <= self.semantic_clearance_gate_deadband < 1.0
            or not np.isfinite(
                self.semantic_planar_follower_min_mobility_scale
            )
            or not 0.0
            < self.semantic_planar_follower_min_mobility_scale
            <= 1.0
        ):
            raise ValueError(
                "B601 planar backpressure/gate/mobility fractions are invalid"
            )
        self.semantic_motion_reverse_deadband = _positive_vector(
            node.declare_parameter(
                "b601_semantic_motion_reverse_deadband_m",
                [0.006, 0.0035, 0.005],
            ).value,
            3,
            "b601_semantic_motion_reverse_deadband_m",
        )
        self.semantic_motion_reverse_confirm_samples = int(
            node.declare_parameter(
                "b601_semantic_motion_reverse_confirm_samples", 3
            ).value
        )
        if not 1 <= self.semantic_motion_reverse_confirm_samples <= 30:
            raise ValueError(
                "b601_semantic_motion_reverse_confirm_samples must be between 1 and 30"
            )
        self.semantic_j1_target_max_velocity = float(
            node.declare_parameter(
                "b601_semantic_j1_target_max_velocity_rad_s", 0.08
            ).value
        )
        self.semantic_j1_target_max_acceleration = float(
            node.declare_parameter(
                "b601_semantic_j1_target_max_acceleration_rad_s2", 0.35
            ).value
        )
        self.semantic_j1_target_deadband = float(
            node.declare_parameter(
                "b601_semantic_j1_target_deadband_rad", 0.0015
            ).value
        )
        if (
            not np.isfinite(self.semantic_j1_target_max_velocity)
            or self.semantic_j1_target_max_velocity <= 0.0
            or not np.isfinite(self.semantic_j1_target_max_acceleration)
            or self.semantic_j1_target_max_acceleration <= 0.0
            or not np.isfinite(self.semantic_j1_target_deadband)
            or self.semantic_j1_target_deadband < 0.0
        ):
            raise ValueError(
                "B601 J1 target velocity/acceleration must be positive and "
                "deadband must be finite and non-negative"
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
                "b601_semantic_j6_target_max_acceleration_rad_s2", 0.40
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
        self.stream_target_q = None
        self.stream_target_velocity = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.stream_target_acceleration = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.target_q_velocity = np.zeros(self.ARM_DOF, dtype=np.float64)
        self.target_q_update_time = 0.0
        self.target_q_update_period_s = self.command_interval_s
        self.last_goal_velocity_hold_s = adaptive_goal_velocity_hold_s(
            self.command_interval_s,
            self.command_interval_s,
            self.stream_goal_velocity_stale_cycles,
            self.stream_goal_velocity_max_hold_s,
        )
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
        self.semantic_j6_neutral_offset_rad = 0.0
        self.semantic_j6_activation_progress = 0.0
        self.semantic_j6_last_raw_angle_rad = None
        self.semantic_j6_stationary_sample_count = 0
        self.semantic_j6_last_palm_offset_m = None
        self.semantic_j6_translation_speed_m_s = 0.0
        self.semantic_j6_raw_innovation_rad = 0.0
        self.semantic_j6_observation_status = "reset"
        self.semantic_vertical_wrist_hold = False
        self.semantic_vertical_wrist_release_count = 0
        self.semantic_vertical_last_wrist_delta_rad = None
        self.semantic_vertical_wrist_rate_rad_s = 0.0
        self.semantic_vertical_last_raw_lift_m = None
        self.semantic_vertical_raw_lift_rate_m_s = 0.0
        self.semantic_vertical_observation_bias_m = 0.0
        self.semantic_vertical_intent_m = 0.0
        self.semantic_vertical_raw_lift_m = 0.0
        self.semantic_vertical_descent_allowed = False
        self.semantic_vertical_descent_guard_row_count = 0
        self.last_semantic_log_time = 0.0
        self.last_semantic_vertical_error_m = 0.0
        self.last_semantic_planar_error_m = np.zeros(2, dtype=np.float64)
        self.last_semantic_forward_gate = 0.0
        self.last_semantic_clearance_scale = 1.0
        self.semantic_planar_command_m = np.zeros(2, dtype=np.float64)
        self.semantic_planar_command_velocity_m_s = np.zeros(2, dtype=np.float64)
        self.semantic_planar_joint_velocity_rad_s = np.zeros(3, dtype=np.float64)
        self.semantic_planar_task_error_vector_m = np.zeros(4, dtype=np.float64)
        self.semantic_planar_backpressure = np.ones(2, dtype=np.float64)
        self.semantic_planar_idle_count = 0
        self.semantic_planar_idle_active = False
        self.semantic_clearance_gate = 0.0
        self.last_semantic_raw_clearance_gate = 0.0
        self.last_semantic_planar_command_time = 0.0
        self.semantic_motion_target_m = np.zeros(3, dtype=np.float64)
        self.semantic_motion_active_direction = np.zeros(3, dtype=np.int8)
        self.semantic_motion_candidate_direction = np.zeros(3, dtype=np.int8)
        self.semantic_motion_candidate_count = np.zeros(3, dtype=np.int64)
        self.semantic_j1_target_velocity = 0.0
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
        self._urdf_joint_lower = self._model.lowerPositionLimit[
            : self.ARM_DOF
        ].copy()
        self._urdf_joint_upper = self._model.upperPositionLimit[
            : self.ARM_DOF
        ].copy()
        self._joint_lower, self._joint_upper = intersect_joint_position_limits(
            self._urdf_joint_lower,
            self._urdf_joint_upper,
            self.configured_safe_joint_lower,
            self.configured_safe_joint_upper,
        )

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
                    "outside the official/URDF joint safety intersection for joints "
                    f"{(np.flatnonzero(outside_hard_limits) + 1).tolist()}"
                )
            self.startup_joint_reference = q.copy()
            self._set_reference_from_q(q)
            self.table_frame_heights = self._frame_heights(q)
            self.target_q = q.copy()
            self.stream_target_q = q.copy()
            self.stream_target_velocity.fill(0.0)
            self.stream_target_acceleration.fill(0.0)
            self.target_q_velocity.fill(0.0)
            self.target_q_update_time = 0.0
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
        safe_lower, safe_upper = joint_tracking_bounds(
            self._joint_lower,
            self._joint_upper,
            self.joint_limit_margin_rad,
            self.startup_joint_reference,
        )
        self.logger.info(
            "B601 official/URDF joint safety envelope after software margin: "
            f"lower={safe_lower.round(4).tolist()}rad, "
            f"upper={safe_upper.round(4).tolist()}rad"
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

    def _semantic_outer_joint_velocity_limits(self) -> np.ndarray:
        """Return the maximum rate at which semantic goals can actually move."""

        limit = self.semantic_max_step / max(self.command_interval_s, 1e-3)
        limit = np.asarray(limit, dtype=np.float64)
        limit[0] = min(limit[0], self.semantic_j1_target_max_velocity)
        limit[3] = min(limit[3], self.semantic_j4_target_max_velocity)
        limit[5] = min(limit[5], self.semantic_j6_target_max_velocity)
        return limit

    def _semantic_joint_bounds(
        self, reference_q: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return semantic offsets clipped by official/URDF tracking limits."""

        reference = _vector(reference_q, self.ARM_DOF, "semantic reference_q")
        tracking_lower, tracking_upper = joint_tracking_bounds(
            self._joint_lower,
            self._joint_upper,
            self.joint_limit_margin_rad,
            self.startup_joint_reference,
        )
        lower = np.maximum(
            reference - self.semantic_max_joint_offset,
            tracking_lower,
        )
        upper = np.minimum(
            reference + self.semantic_max_joint_offset,
            tracking_upper,
        )
        if np.any(lower > upper):
            raise RuntimeError("B601 semantic joint bounds have no valid overlap")
        return lower, upper

    def _control_loop(self, robot: Any, dt: float) -> None:
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
                nominal_dt = 1.0 / self.control_rate_hz
                trajectory_dt = (
                    nominal_dt
                    if not np.isfinite(dt) or float(dt) <= 0.0
                    else float(np.clip(float(dt), 0.25 * nominal_dt, 2.0 * nominal_dt))
                )
                with self.target_lock:
                    goal_q = q.copy() if self.target_q is None else self.target_q.copy()
                    if self.stream_target_q is None:
                        self.stream_target_q = q.copy()
                        self.stream_target_velocity.fill(0.0)
                        self.stream_target_acceleration.fill(0.0)
                    if self.home_motion_active:
                        # Table-home already supplies a verified minimum-jerk
                        # path.  Do not stack a second trajectory on it or its
                        # final verification can time out behind the command.
                        stream_q = goal_q
                        stream_velocity = np.zeros(self.ARM_DOF, dtype=np.float64)
                        stream_acceleration = np.zeros(self.ARM_DOF, dtype=np.float64)
                    else:
                        stream_velocity_limit = self.stream_max_velocity.copy()
                        stream_acceleration_limit = self.stream_max_acceleration.copy()
                        goal_velocity = np.zeros(self.ARM_DOF, dtype=np.float64)
                        stream_correction_gain = None
                        stream_correction_limit = None
                        if self.control_mode == "semantic_joint":
                            outer_velocity_limit = (
                                self._semantic_outer_joint_velocity_limits()
                            )
                            stream_follow_scale = np.full(
                                self.ARM_DOF,
                                self.stream_outer_velocity_scale,
                                dtype=np.float64,
                            )
                            stream_follow_scale[1:4] = (
                                self.stream_planar_outer_velocity_scale
                            )
                            stream_velocity_limit = coupled_stream_velocity_limits(
                                stream_velocity_limit,
                                self.semantic_max_step,
                                self.command_interval_s,
                                outer_velocity_limit,
                                stream_follow_scale,
                            )
                            stream_correction_gain = (
                                self.stream_position_correction_gain_s
                            )
                            correction_fraction = np.full(
                                self.ARM_DOF,
                                self.stream_position_correction_max_fraction,
                                dtype=np.float64,
                            )
                            correction_fraction[1:4] = (
                                self.stream_planar_position_correction_max_fraction
                            )
                            stream_correction_limit = (
                                correction_fraction * stream_velocity_limit
                            )
                            goal_age = time.monotonic() - self.target_q_update_time
                            goal_hold_s = adaptive_goal_velocity_hold_s(
                                self.command_interval_s,
                                self.target_q_update_period_s,
                                self.stream_goal_velocity_stale_cycles,
                                self.stream_goal_velocity_max_hold_s,
                            )
                            self.last_goal_velocity_hold_s = goal_hold_s
                            if (
                                self.target_q_update_time > 0.0
                                and goal_age <= goal_hold_s
                            ):
                                goal_velocity = np.clip(
                                    self.target_q_velocity,
                                    -stream_velocity_limit,
                                    stream_velocity_limit,
                                )
                        (
                            stream_q,
                            stream_velocity,
                            stream_acceleration,
                        ) = jerk_limited_joint_target(
                            goal_q,
                            self.stream_target_q,
                            self.stream_target_velocity,
                            self.stream_target_acceleration,
                            stream_velocity_limit,
                            stream_acceleration_limit,
                            self.stream_max_jerk,
                            trajectory_dt,
                            requested_velocity=goal_velocity,
                            position_correction_gain_s=stream_correction_gain,
                            position_correction_max_velocity=stream_correction_limit,
                        )
                    self.stream_target_q = stream_q
                    self.stream_target_velocity = stream_velocity
                    self.stream_target_acceleration = stream_acceleration
                    target_q = stream_q.copy()
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
            self.stream_target_q = np.asarray(q, dtype=np.float64).copy()
            self.stream_target_velocity.fill(0.0)
            self.stream_target_acceleration.fill(0.0)
            self.target_q_velocity.fill(0.0)
            self.target_q_update_time = 0.0
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
        self.semantic_j6_neutral_offset_rad = 0.0
        self.semantic_j6_activation_progress = 0.0
        self.semantic_j6_last_raw_angle_rad = None
        self.semantic_j6_stationary_sample_count = 0
        self.semantic_j6_last_palm_offset_m = None
        self.semantic_j6_translation_speed_m_s = 0.0
        self.semantic_j6_raw_innovation_rad = 0.0
        self.semantic_j6_observation_status = "reset"
        self.semantic_vertical_wrist_hold = False
        self.semantic_vertical_wrist_release_count = 0
        self.semantic_vertical_last_wrist_delta_rad = None
        self.semantic_vertical_wrist_rate_rad_s = 0.0
        self.semantic_vertical_last_raw_lift_m = None
        self.semantic_vertical_raw_lift_rate_m_s = 0.0
        self.semantic_vertical_observation_bias_m = 0.0
        self.semantic_vertical_intent_m = 0.0
        self.semantic_vertical_raw_lift_m = 0.0
        self.semantic_vertical_descent_allowed = False
        self.semantic_vertical_descent_guard_row_count = 0

    def _reset_semantic_smoothing(self, now_s: float = 0.0) -> None:
        self.semantic_planar_command_m.fill(0.0)
        self.semantic_planar_command_velocity_m_s.fill(0.0)
        self.semantic_planar_joint_velocity_rad_s.fill(0.0)
        self.semantic_planar_task_error_vector_m.fill(0.0)
        self.semantic_planar_backpressure.fill(1.0)
        self.semantic_planar_idle_count = 0
        self.semantic_planar_idle_active = False
        self.semantic_clearance_gate = 0.0
        self.last_semantic_raw_clearance_gate = 0.0
        self.target_q_update_period_s = self.command_interval_s
        self.last_goal_velocity_hold_s = adaptive_goal_velocity_hold_s(
            self.command_interval_s,
            self.command_interval_s,
            self.stream_goal_velocity_stale_cycles,
            self.stream_goal_velocity_max_hold_s,
        )
        self.last_semantic_planar_command_time = float(now_s)
        self.semantic_motion_target_m.fill(0.0)
        self.semantic_motion_active_direction.fill(0)
        self.semantic_motion_candidate_direction.fill(0)
        self.semantic_motion_candidate_count.fill(0)
        self.semantic_j1_target_velocity = 0.0
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
            self.stream_target_q = q.copy()
            self.stream_target_velocity.fill(0.0)
            self.stream_target_acceleration.fill(0.0)
            self.target_q_velocity.fill(0.0)
            self.target_q_update_time = 0.0
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
                self.stream_target_q = q.copy()
                self.stream_target_velocity.fill(0.0)
                self.stream_target_acceleration.fill(0.0)
                self.target_q_velocity.fill(0.0)
                self.target_q_update_time = 0.0
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
        with self.target_lock:
            self.target_q = q_reference.copy()
            self.target_q_velocity.fill(0.0)
            self.target_q_update_time = 0.0
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
        target_dt = (
            self.command_interval_s
            if self.last_submit_time <= 0.0
            else float(np.clip(now - self.last_submit_time, 0.001, 0.2))
        )
        q_reference = self.semantic_control_reference_q
        with self.target_lock:
            previous_target = (
                q_seed.copy()
                if self.target_q is None
                else self.target_q.copy()
            )

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
                self.stream_target_q = q_seed.copy()
                self.stream_target_velocity.fill(0.0)
                self.stream_target_acceleration.fill(0.0)
                self.target_q_velocity.fill(0.0)
                self.target_q_update_time = 0.0
            self._reset_semantic_smoothing(now)
            self._reset_filters()
            if now - self.last_semantic_log_time >= 1.0:
                self.logger.warn(
                    "Holding B601 measured pose: semantic palm displacement "
                    f"is implausible ({camera_offset.round(3).tolist()}m)"
                )
                self.last_semantic_log_time = now
            return self.last_arm_offset
        camera_offset = self._anti_drift_filter_offset(camera_offset)
        if self.semantic_j6_last_palm_offset_m is None:
            self.semantic_j6_translation_speed_m_s = 0.0
        else:
            self.semantic_j6_translation_speed_m_s = float(
                np.linalg.norm(
                    camera_offset - self.semantic_j6_last_palm_offset_m
                )
                / target_dt
            )
        self.semantic_j6_last_palm_offset_m = camera_offset.copy()

        raw_wrist_twist = observation.palm_roll_rad
        raw_wrist_valid = bool(
            raw_wrist_twist is not None and np.isfinite(raw_wrist_twist)
        )
        raw_wrist_accepted = False
        raw_innovation = 0.0
        if raw_wrist_valid:
            raw_wrist_accepted, raw_innovation = (
                validate_wrapped_angle_observation(
                    float(raw_wrist_twist),
                    self.semantic_j6_last_raw_angle_rad,
                    target_dt,
                    self.wrist_twist_max_raw_rate_rad_s,
                    self.wrist_twist_max_raw_jump_rad,
                )
            )
            self.semantic_j6_raw_innovation_rad = raw_innovation
            if raw_wrist_accepted:
                self.semantic_j6_last_raw_angle_rad = float(raw_wrist_twist)
            else:
                # Rebase only after a new stable sequence. Keeping the old
                # raw sample here could reject every frame after a genuine
                # large occlusion/reacquisition change.
                self.semantic_j6_last_raw_angle_rad = None
        else:
            self.semantic_j6_raw_innovation_rad = 0.0
            self.semantic_j6_last_raw_angle_rad = None

        translation_stable = bool(
            self.semantic_j6_translation_speed_m_s
            <= self.wrist_twist_translation_gate_m_s
        )
        if raw_wrist_accepted and translation_stable:
            self.semantic_j6_stationary_sample_count += 1
        else:
            self.semantic_j6_stationary_sample_count = 0
        wrist_observation_ready = bool(
            raw_wrist_accepted
            and translation_stable
            and self.semantic_j6_stationary_sample_count
            >= self.wrist_twist_stationary_confirm_samples
        )
        if not raw_wrist_valid:
            self.semantic_j6_observation_status = "invalid_hold"
        elif not raw_wrist_accepted:
            self.semantic_j6_observation_status = "outlier_hold"
        elif not translation_stable:
            self.semantic_j6_observation_status = "translation_hold"
        elif not wrist_observation_ready:
            self.semantic_j6_observation_status = "stabilizing"
        else:
            self.semantic_j6_observation_status = "ready"
        raw_lateral = self._scalar_deadband(
            camera_offset[0], self.semantic_linear_deadband_m[0]
        )
        # Camera Z increases away from the camera; human forward motion is -Z.
        raw_forward = self._scalar_deadband(
            -camera_offset[2], self.semantic_linear_deadband_m[1]
        )
        raw_forward *= self.semantic_planar_input_scale[0]
        raw_forward = float(
            np.clip(
                raw_forward,
                -self.semantic_planar_max_forward_m,
                self.semantic_planar_max_forward_m,
            )
        )
        # Keep the unbounded signed observation until wrist-rotation
        # compensation is applied. Clipping it to zero first would discard the
        # information needed to recognize a later deliberate downward move.
        raw_lift_unbounded = self._scalar_deadband(
            -camera_offset[1], self.semantic_linear_deadband_m[2]
        )
        raw_lift_unbounded *= self.semantic_planar_input_scale[1]
        self.semantic_vertical_raw_lift_m = float(raw_lift_unbounded)
        if self.semantic_vertical_last_raw_lift_m is None:
            self.semantic_vertical_raw_lift_rate_m_s = 0.0
        else:
            self.semantic_vertical_raw_lift_rate_m_s = float(
                (raw_lift_unbounded - self.semantic_vertical_last_raw_lift_m)
                / target_dt
            )
        self.semantic_vertical_last_raw_lift_m = float(raw_lift_unbounded)

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
        raw_forward_gate = clearance_fraction**2 * (
            3.0 - 2.0 * clearance_fraction
        )
        forward_gate = rate_limited_unit_interval(
            raw_forward_gate,
            self.semantic_clearance_gate,
            self.semantic_clearance_gate_rates[0],
            self.semantic_clearance_gate_rates[1],
            target_dt,
            self.semantic_clearance_gate_deadband,
        )
        # Hysteresis must never delay a near-table lockout. The rate limit is
        # used only away from the final safety boundary; the modeled table
        # projection below remains the independent fail-closed guard.
        if clearance_fraction <= 0.10 and raw_forward_gate < forward_gate:
            forward_gate = raw_forward_gate
        self.last_semantic_raw_clearance_gate = raw_forward_gate
        self.semantic_clearance_gate = forward_gate
        self.last_semantic_forward_gate = forward_gate
        articulation_ready = forward_gate >= 0.95
        # Gross palm translation changes the human arm-plane reference and a
        # bad depth frame can flip the palm normal.  Either condition disarms
        # wrist intent and freezes the last accepted J6 target.  Re-arming
        # requires several stable frames and captures a fresh neutral angle,
        # so translation cannot accumulate into a false wrist command.
        must_hold_j6 = bool(
            clearance_fraction <= 0.50
            or not raw_wrist_accepted
            or not translation_stable
        )
        if self.semantic_j6_clearance_enabled and must_hold_j6:
            held_offset = float(
                np.clip(
                    previous_target[5] - q_reference[5],
                    -self.semantic_max_joint_offset[5],
                    self.semantic_max_joint_offset[5],
                )
            )
            self.semantic_j6_clearance_enabled = False
            self.semantic_j6_active_direction = 0
            self.semantic_j6_held_delta_rad = 0.0
            self.semantic_j6_candidate_direction = 0
            self.semantic_j6_candidate_count = 0
            self.semantic_j6_activation_progress = 0.0
            self.semantic_j6_neutral_offset_rad = held_offset
            self.semantic_j6_command_offset_rad = held_offset
            self.semantic_signal_references["wrist_twist"] = None
            self.semantic_signal_deltas["wrist_twist"] = 0.0
            if clearance_fraction <= 0.50:
                self.semantic_j6_observation_status = "clearance_hold"

        if (
            not self.semantic_j6_clearance_enabled
            and clearance_fraction >= 0.95
            and wrist_observation_ready
        ):
            self.semantic_j6_clearance_enabled = True
            self.semantic_signal_references["wrist_twist"] = float(
                raw_wrist_twist
            )
            self.semantic_signal_deltas["wrist_twist"] = 0.0
            self.semantic_j6_active_direction = 0
            self.semantic_j6_held_delta_rad = 0.0
            self.semantic_j6_candidate_direction = 0
            self.semantic_j6_candidate_count = 0
            self.semantic_j6_activation_progress = 0.0
            self.semantic_j6_neutral_offset_rad = float(
                np.clip(
                    previous_target[5] - q_reference[5],
                    -self.semantic_max_joint_offset[5],
                    self.semantic_max_joint_offset[5],
                )
            )
            self.semantic_j6_command_offset_rad = (
                self.semantic_j6_neutral_offset_rad
            )

        wrist_intent_update_allowed = bool(
            self.semantic_j6_clearance_enabled and wrist_observation_ready
        )
        roll_delta = float(self.semantic_signal_deltas["wrist_twist"])
        if wrist_intent_update_allowed:
            roll_delta = self._semantic_angle_delta(
                "wrist_twist",
                float(raw_wrist_twist),
                self.semantic_angle_deadband_rad[2],
            )
        previous_j6_direction = self.semantic_j6_active_direction
        if wrist_intent_update_allowed:
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
                self.wrist_twist_reverse_confirm_samples,
            )
        if not wrist_intent_update_allowed:
            self.semantic_j6_activation_progress = 0.0
        elif self.semantic_j6_active_direction == 0:
            self.semantic_j6_activation_progress = 0.0
        elif self.semantic_j6_active_direction != previous_j6_direction:
            self.semantic_j6_activation_progress = min(
                1.0,
                target_dt / self.wrist_twist_soft_start_s,
            )
        else:
            self.semantic_j6_activation_progress = min(
                1.0,
                self.semantic_j6_activation_progress
                + target_dt / self.wrist_twist_soft_start_s,
            )

        vertical_decoupling_active = bool(
            self.vertical_wrist_decoupling_enabled
            and self.semantic_channel_enable[4] > 0.5
            and clearance_fraction >= 0.95
        )
        vertical_wrist_armed = bool(
            self.semantic_j6_active_direction != 0
            or abs(roll_delta) >= self.vertical_wrist_hold_min_delta_rad
        )
        if vertical_decoupling_active:
            # Use the faster J6 angle filter (alpha 0.18 in the validated
            # profile), not raw angular velocity: hardware logs show raw pose
            # jitter above 8 deg/s even with no confirmed wrist gesture. Once
            # latched, height settling is an independent release condition.
            vertical_release_observation_valid = wrist_height_release_ready(
                raw_wrist_valid and raw_wrist_accepted,
                self.semantic_vertical_wrist_hold,
                self.semantic_vertical_raw_lift_rate_m_s,
                self.vertical_wrist_release_lift_rate_m_s,
            )
            (
                self.semantic_vertical_wrist_hold,
                self.semantic_vertical_wrist_release_count,
                self.semantic_vertical_wrist_rate_rad_s,
            ) = wrapped_rate_motion_hold(
                roll_delta,
                self.semantic_vertical_last_wrist_delta_rad,
                target_dt,
                self.semantic_vertical_wrist_hold,
                self.semantic_vertical_wrist_release_count,
                self.vertical_wrist_hold_rate_rad_s,
                self.vertical_wrist_release_rate_rad_s,
                self.vertical_wrist_release_samples,
                vertical_release_observation_valid,
                activation_enabled=vertical_wrist_armed,
            )
        else:
            self.semantic_vertical_wrist_hold = False
            self.semantic_vertical_wrist_release_count = 0
            self.semantic_vertical_wrist_rate_rad_s = 0.0
            if self.semantic_channel_enable[4] <= 0.5:
                self.semantic_vertical_observation_bias_m = 0.0
        self.semantic_vertical_last_wrist_delta_rad = roll_delta
        (
            compensated_lift,
            self.semantic_vertical_observation_bias_m,
        ) = rotation_compensated_scalar_observation(
            raw_lift_unbounded,
            self.semantic_vertical_intent_m,
            self.semantic_vertical_observation_bias_m,
            self.semantic_vertical_wrist_hold,
            0.0,
            self.semantic_vertical_max_lift_m,
        )
        self.semantic_vertical_intent_m = compensated_lift

        raw_motion_target = np.asarray(
            [raw_lateral, raw_forward, compensated_lift],
            dtype=np.float64,
        )
        (
            stable_motion_target,
            next_motion_active_direction,
            next_motion_candidate_direction,
            next_motion_candidate_count,
        ) = confirm_vector_reversals(
            raw_motion_target,
            self.semantic_motion_target_m,
            self.semantic_motion_active_direction,
            self.semantic_motion_candidate_direction,
            self.semantic_motion_candidate_count,
            self.semantic_motion_reverse_deadband,
            self.semantic_motion_reverse_confirm_samples,
        )
        lateral, forward, lift = [float(value) for value in stable_motion_target]

        direct_offsets = semantic_direct_joint_offsets(
            lateral,
            self.semantic_j6_held_delta_rad,
            self.semantic_joint_gains,
            self.semantic_channel_enable,
            self.semantic_max_joint_offset,
        )
        if self.semantic_channel_enable[4] <= 0.5:
            direct_offsets[5] = 0.0
        elif not wrist_intent_update_allowed:
            # Freeze the accepted outer target, not the full unachieved vision
            # offset.  A rejected frame therefore cannot keep J6 travelling.
            direct_offsets[5] = float(
                np.clip(
                    previous_target[5] - q_reference[5],
                    -self.semantic_max_joint_offset[5],
                    self.semantic_max_joint_offset[5],
                )
            )
        elif (
            self.semantic_j6_active_direction != 0
            and abs(float(direct_offsets[5])) > 1e-9
        ):
            full_j6_contribution = float(
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
                    self.semantic_j6_neutral_offset_rad
                    + minimum_jerk_blend(
                        self.semantic_j6_activation_progress
                    )
                    * full_j6_contribution,
                    -self.semantic_max_joint_offset[5],
                    self.semantic_max_joint_offset[5],
                )
            )
        else:
            direct_offsets[5] = self.semantic_j6_neutral_offset_rad
        self.semantic_j6_command_offset_rad = float(direct_offsets[5])
        direct_offsets[0] = gate_outward_scalar_increment(
            direct_offsets[0],
            previous_target[0] - q_reference[0],
            forward_gate,
        )
        semantic_lower, semantic_upper = self._semantic_joint_bounds(q_reference)
        # Saturate direct J1/J6 requests at the safe envelope instead of
        # rejecting the complete J2/J3/J4 update when the operator reaches an
        # expanded-workspace boundary.
        q_base = np.clip(
            q_reference + direct_offsets,
            semantic_lower,
            semantic_upper,
        )
        requested_forward_target = (
            forward
            if self.semantic_channel_enable[1] > 0.5
            else 0.0
        )
        requested_lift_target = (
            lift if self.semantic_channel_enable[5] > 0.5 else 0.0
        )
        previous_planar_velocity = (
            self.semantic_planar_command_velocity_m_s.copy()
        )
        (
            requested_lift_target,
            previous_planar_velocity[1],
        ) = wrist_hold_vertical_planner_state(
            requested_lift_target,
            self.semantic_planar_command_m[1],
            previous_planar_velocity[1],
            self.semantic_vertical_wrist_hold,
        )
        requested_planar_target = np.asarray(
            [requested_forward_target, requested_lift_target],
            dtype=np.float64,
        )
        (
            self.semantic_planar_idle_active,
            self.semantic_planar_idle_count,
        ) = confirmed_stationary_planar_target(
            requested_planar_target,
            self.semantic_planar_command_m,
            previous_planar_velocity,
            self.semantic_planar_idle_target_deadband,
            self.semantic_planar_idle_velocity_deadband,
            self.semantic_planar_idle_count,
            self.semantic_planar_idle_confirm_samples,
        )
        if self.semantic_planar_idle_active:
            # Keep the last accepted task target and brake the coordinated
            # joint velocity.  The 500 Hz MIT loop still finishes and holds
            # its already accepted setpoint; only redundant task re-solving
            # from stationary visual noise is suppressed.
            requested_planar_target = self.semantic_planar_command_m.copy()
            previous_planar_velocity.fill(0.0)
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
        candidate_planar_command, _candidate_planar_velocity = (
            acceleration_limited_vector_target(
                requested_planar_target,
                self.semantic_planar_command_m,
                previous_planar_velocity,
                self.semantic_planar_command_max_velocity,
                self.semantic_planar_command_max_acceleration,
                self.semantic_planar_command_deadband,
                planar_dt,
            )
        )
        candidate_planar_command[0] = gate_outward_scalar_increment(
            candidate_planar_command[0],
            self.semantic_planar_command_m[0],
            forward_gate,
        )
        planar_increment = candidate_planar_command - self.semantic_planar_command_m
        task_tracking_limit = np.asarray(
            [
                self.semantic_planar_tracking_error[0],
                *([self.semantic_planar_tracking_error[1]] * 3),
            ],
            dtype=np.float64,
        )
        raw_backpressure = independent_planar_task_backpressure(
            self.semantic_planar_task_error_vector_m,
            planar_increment,
            self.semantic_vertical_height_profile,
            task_tracking_limit,
            self.semantic_planar_backpressure_start_fraction,
        )
        planar_backpressure = np.asarray(
            [
                rate_limited_unit_interval(
                    raw_backpressure[index],
                    self.semantic_planar_backpressure[index],
                    self.semantic_planar_backpressure_rates[0],
                    self.semantic_planar_backpressure_rates[1],
                    planar_dt,
                )
                for index in range(2)
            ],
            dtype=np.float64,
        )
        planar_command = (
            self.semantic_planar_command_m
            + planar_backpressure * planar_increment
        )
        planar_velocity = (
            planar_command - self.semantic_planar_command_m
        ) / planar_dt
        forward_target = float(planar_command[0])
        lift_target = float(planar_command[1])
        (
            q_desired,
            requested_planar_joint_velocity,
            next_planar_task_error,
        ) = self._resolve_semantic_planar_rate(
            q_base,
            q_seed,
            previous_target,
            forward_target,
            lift_target,
            planar_velocity,
            forward_gate,
            planar_dt,
            settle_hold=self.semantic_planar_idle_active,
        )
        resolved_max_step = self.semantic_max_step.copy()
        planar_indices = np.asarray([1, 2, 3], dtype=np.int64)
        resolved_max_step[planar_indices] = (
            self._semantic_outer_joint_velocity_limits()[planar_indices]
            * planar_dt
        )
        # J1/J6 have dedicated velocity/acceleration limiters below. Feeding
        # their already step-limited values into those limiters made every
        # small upstream step look immediately reachable, so the acceleration
        # state repeatedly collapsed to zero. Exclude them from the generic
        # ramp and limit the full direct-joint request exactly once.
        ramp_desired = q_desired.copy()
        ramp_desired[[0, 5]] = previous_target[[0, 5]]
        q_target = ramp_joint_target_with_tracking_window(
            ramp_desired,
            q_seed,
            previous_target,
            resolved_max_step,
            self.semantic_max_tracking_error,
            coordinated_indices=(1, 2, 3),
        )
        self.last_semantic_vertical_primary_progress = clearance_fraction
        self.last_semantic_vertical_follower_limit = (
            self.semantic_planar_follower_min_mobility_scale
            + (1.0 - self.semantic_planar_follower_min_mobility_scale)
            * forward_gate
        )
        (
            j1_target,
            j1_velocity,
        ) = acceleration_limited_scalar_target_with_tracking_window(
            float(q_desired[0]),
            float(q_seed[0]),
            float(previous_target[0]),
            self.semantic_j1_target_velocity,
            self.semantic_j1_target_max_velocity,
            self.semantic_j1_target_max_acceleration,
            self.semantic_j1_target_deadband,
            float(self.semantic_max_tracking_error[0]),
            target_dt,
        )
        q_target[0] = j1_target
        (
            j6_target,
            j6_velocity,
        ) = acceleration_limited_scalar_target_with_tracking_window(
            float(q_desired[5]),
            float(q_seed[5]),
            float(previous_target[5]),
            self.semantic_j6_target_velocity,
            self.semantic_j6_target_max_velocity,
            self.semantic_j6_target_max_acceleration,
            self.semantic_j6_target_deadband,
            float(self.semantic_max_tracking_error[5]),
            target_dt,
        )
        q_target[5] = j6_target
        q_target, clearance_scale = self._project_semantic_target_to_table_clearance(
            q_seed,
            q_target,
        )
        self.last_semantic_clearance_scale = clearance_scale
        if clearance_scale < 1.0:
            planar_command = (
                self.semantic_planar_command_m
                + clearance_scale
                * (planar_command - self.semantic_planar_command_m)
            )
            planar_velocity = planar_velocity * clearance_scale
        accepted_planar_joint_velocity = (
            q_target[planar_indices] - previous_target[planar_indices]
        ) / planar_dt
        j1_velocity = float(
            np.clip(
                (q_target[0] - previous_target[0]) / target_dt,
                -self.semantic_j1_target_max_velocity,
                self.semantic_j1_target_max_velocity,
            )
        )
        j6_velocity = float(
            np.clip(
                (q_target[5] - previous_target[5]) / target_dt,
                -self.semantic_j6_target_max_velocity,
                self.semantic_j6_target_max_velocity,
            )
        )
        j4_velocity = float(accepted_planar_joint_velocity[2])
        if clearance_scale < 1.0 and now - self.last_semantic_log_time >= 1.0:
            self.logger.warn(
                "Clamped B601 semantic target at the validated table-height "
                f"boundary (safe_progress={clearance_scale:.3f})"
            )
            self.last_semantic_log_time = now
        if not self._semantic_table_clearance_ok(q_seed, q_target):
            # This is a fail-closed guard for model/numerical errors.  Normal
            # boundary handling is performed by the projection above and does
            # not discard the complete semantic update.
            if now - self.last_semantic_log_time >= 1.0:
                self.logger.warn(
                    "Refusing B601 semantic target: a modeled frame would move "
                    "below its validated table-pose height"
                )
                self.last_semantic_log_time = now
            with self.target_lock:
                self.target_q_velocity.fill(0.0)
                self.target_q_update_time = 0.0
            self.semantic_planar_joint_velocity_rad_s.fill(0.0)
            self.semantic_planar_command_velocity_m_s.fill(0.0)
            return self.last_arm_offset
        if not self._joint_target_is_safe(q_target, q_seed):
            self.last_semantic_planar_command_time = now
            self.semantic_planar_command_velocity_m_s.fill(0.0)
            self.semantic_j1_target_velocity = 0.0
            self.semantic_j4_target_velocity = 0.0
            self.semantic_j6_target_velocity = 0.0
            self.semantic_planar_joint_velocity_rad_s.fill(0.0)
            with self.target_lock:
                self.target_q_velocity.fill(0.0)
                self.target_q_update_time = 0.0
            return self.last_arm_offset

        outer_velocity_limit = self._semantic_outer_joint_velocity_limits()
        with self.target_lock:
            if self.target_q_update_time > 0.0:
                observed_period = now - self.target_q_update_time
                if (
                    np.isfinite(observed_period)
                    and 0.5 * self.command_interval_s
                    <= observed_period
                    <= 2.0 * self.stream_goal_velocity_max_hold_s
                ):
                    alpha = self.stream_goal_period_filter_alpha
                    bounded_period = min(
                        observed_period,
                        self.stream_goal_velocity_max_hold_s / 1.5,
                    )
                    if bounded_period > self.target_q_update_period_s:
                        # Adapt immediately when vision is slower so velocity
                        # feed-forward does not expire between the next frames.
                        self.target_q_update_period_s = bounded_period
                    else:
                        self.target_q_update_period_s = (
                            alpha * bounded_period
                            + (1.0 - alpha) * self.target_q_update_period_s
                        )
            next_goal_velocity = smoothed_semantic_goal_velocity(
                q_target,
                previous_target,
                self.target_q_velocity,
                outer_velocity_limit,
                self.stream_goal_velocity_filter_alpha,
                target_dt,
                j1_velocity,
                j6_velocity,
            )
            self.target_q_velocity = next_goal_velocity
            self.target_q = q_target.copy()
            self.target_q_update_time = now
        self.semantic_planar_command_m = planar_command
        self.semantic_planar_command_velocity_m_s = planar_velocity
        self.semantic_planar_joint_velocity_rad_s = (
            accepted_planar_joint_velocity.copy()
        )
        self.semantic_planar_task_error_vector_m = next_planar_task_error.copy()
        self.semantic_planar_backpressure = planar_backpressure
        self.last_semantic_planar_command_time = now
        self.semantic_motion_target_m = stable_motion_target
        self.semantic_motion_active_direction = next_motion_active_direction
        self.semantic_motion_candidate_direction = next_motion_candidate_direction
        self.semantic_motion_candidate_count = next_motion_candidate_count
        self.semantic_j1_target_velocity = j1_velocity
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
                f"side={observation.side}, "
                f"raw_motion={raw_motion_target.round(4).tolist()}m, "
                f"lateral={lateral:+.4f}m, "
                f"forward={forward:+.4f}m, planar_forward={forward_target:+.4f}m, "
                f"lift={lift_target:+.4f}m, "
                f"raw_lift={self.semantic_vertical_raw_lift_m:+.4f}m, "
                f"lift_bias={self.semantic_vertical_observation_bias_m:+.4f}m, "
                f"vertical_wrist_armed={vertical_wrist_armed}, "
                f"vertical_wrist_hold={self.semantic_vertical_wrist_hold}, "
                f"vertical_wrist_rate="
                f"{np.rad2deg(self.semantic_vertical_wrist_rate_rad_s):+.1f}deg/s, "
                f"vertical_lift_rate="
                f"{self.semantic_vertical_raw_lift_rate_m_s:+.3f}m/s, "
                f"vertical_wrist_release="
                f"{self.semantic_vertical_wrist_release_count}/"
                f"{self.vertical_wrist_release_samples}, "
                f"vertical_descent_allowed="
                f"{self.semantic_vertical_descent_allowed}, "
                f"descent_guard_rows="
                f"{self.semantic_vertical_descent_guard_row_count}/"
                f"{len(self._articulation_clearance_frame_ids)}, "
                f"forward_gate={forward_gate:.2f}/{raw_forward_gate:.2f}, "
                f"backpressure={planar_backpressure.round(2).tolist()}/"
                f"{raw_backpressure.round(2).tolist()}, "
                f"planar_idle={self.semantic_planar_idle_active}:"
                f"{self.semantic_planar_idle_count}/"
                f"{self.semantic_planar_idle_confirm_samples}, "
                f"planar_velocity={planar_velocity.round(4).tolist()}m/s, "
                f"resolved_qdot={accepted_planar_joint_velocity.round(4).tolist()}rad/s, "
                f"requested_qdot={requested_planar_joint_velocity.round(4).tolist()}rad/s, "
                f"reverse_pending={next_motion_candidate_direction.tolist()}/"
                f"{next_motion_candidate_count.tolist()}, "
                f"tcp_lift={tcp_lift:+.4f}m, "
                f"chain_clearance={chain_clearance:+.4f}m, "
                f"solver_error=[forward={self.last_semantic_planar_error_m[0]:+.4f}, "
                f"height={self.last_semantic_planar_error_m[1]:+.4f}]m, "
                f"actual_error=[forward={actual_planar_error[0]:+.4f}, "
                f"height={actual_planar_error[1]:+.4f}]m, "
                f"clearance_fraction={self.last_semantic_vertical_primary_progress:.2f}, "
                f"follower_mobility={self.last_semantic_vertical_follower_limit:.2f}, "
                f"clearance_scale={self.last_semantic_clearance_scale:.3f}, "
                f"J1_target_velocity={self.semantic_j1_target_velocity:+.4f}rad/s, "
                f"J4_target_velocity={self.semantic_j4_target_velocity:+.4f}rad/s, "
                f"J6_target_velocity={self.semantic_j6_target_velocity:+.4f}rad/s, "
                f"goal_velocity={self.target_q_velocity.round(4).tolist()}rad/s, "
                f"goal_period={self.target_q_update_period_s:.3f}s, "
                f"goal_hold={self.last_goal_velocity_hold_s:.3f}s, "
                f"stream_velocity={self.stream_target_velocity.round(4).tolist()}rad/s, "
                f"articulation={'ready' if articulation_ready else 'locked'}, "
                f"J6_clearance={'armed' if self.semantic_j6_clearance_enabled else 'locked'}, "
                f"wrist_twist={wrist_twist_text}, "
                f"J6_observation={self.semantic_j6_observation_status}, "
                f"J6_raw_innovation="
                f"{np.rad2deg(self.semantic_j6_raw_innovation_rad):+.1f}deg, "
                f"J6_translation_speed="
                f"{self.semantic_j6_translation_speed_m_s:.3f}m/s, "
                f"J6_stable={self.semantic_j6_stationary_sample_count}/"
                f"{self.wrist_twist_stationary_confirm_samples}, "
                f"J6_delta={np.rad2deg(roll_delta):+.1f}deg, "
                f"J6_confirmed={np.rad2deg(self.semantic_j6_held_delta_rad):+.1f}deg, "
                f"J6_direction={self.semantic_j6_active_direction:+d}, "
                f"J6_activation={self.semantic_j6_activation_progress:.2f}, "
                f"J6_pending={self.semantic_j6_candidate_count}/"
                f"{self.wrist_twist_confirm_samples}:"
                f"{self.wrist_twist_reverse_confirm_samples}, "
                f"J6_neutral="
                f"{np.rad2deg(self.semantic_j6_neutral_offset_rad):+.1f}deg, "
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

    def _resolve_semantic_planar_rate(
        self,
        q_base: np.ndarray,
        q_measured: np.ndarray,
        previous_target: np.ndarray,
        forward_m: float,
        lift_m: float,
        planar_velocity_m_s: np.ndarray,
        clearance_gate: float,
        dt_s: float,
        settle_hold: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve a continuous J2/J3/J4 velocity from the planar task.

        The old absolute IK path produced a distant ``q_desired`` and then
        chased it through a per-frame ramp.  Here the task error and task
        feed-forward velocity are resolved at measured feedback, acceleration
        limited as one coordinated joint vector, and integrated only for this
        outer-loop interval.
        """

        q_base = np.asarray(q_base, dtype=np.float64).copy()
        q_measured = np.asarray(q_measured, dtype=np.float64).copy()
        previous_target = np.asarray(previous_target, dtype=np.float64).copy()
        task_velocity = np.asarray(planar_velocity_m_s, dtype=np.float64).reshape(2)
        dt = float(dt_s)
        columns = np.flatnonzero(self.semantic_vertical_joint_mask > 0.5)
        q_request = previous_target.copy()
        direct_columns = np.setdiff1d(
            np.arange(self.ARM_DOF, dtype=np.int64), columns
        )
        q_request[direct_columns] = q_base[direct_columns]

        base_positions = self._frame_positions(q_base)
        base_tcp = base_positions[self._end_frame_id]
        try:
            forward_axis, base_forward = horizontal_radial_axis(base_tcp)
        except ValueError:
            self.logger.warn(
                "B601 TCP is too close to the base axis for resolved-rate control; holding"
            )
            return q_request, np.zeros(columns.size), np.zeros(4)

        q_model = self._pad_q_for_model(self._model, q_measured, self.ARM_DOF)
        self._pin.forwardKinematics(self._model, self._kinematics_data, q_model)
        self._pin.updateFramePlacements(self._model, self._kinematics_data)
        current_tcp = np.asarray(
            self._kinematics_data.oMf[self._end_frame_id].translation,
            dtype=np.float64,
        )
        current_heights = np.asarray(
            [
                float(self._kinematics_data.oMf[frame_id].translation[2])
                for frame_id in self._articulation_clearance_frame_ids
            ],
            dtype=np.float64,
        )
        target_forward = base_forward + float(forward_m)
        target_heights = (
            base_positions[self._articulation_clearance_frame_ids, 2]
            + self.semantic_vertical_height_profile * float(lift_m)
        )
        task_error = np.concatenate(
            (
                [
                    target_forward
                    - float(np.dot(current_tcp[:2], forward_axis))
                ],
                target_heights - current_heights,
            )
        )

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
            raise RuntimeError("B601 resolved-rate task is missing the TCP frame")
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
                self.semantic_vertical_orientation_weight
                * end_jacobian[3:6, :],
            )
        )

        error_gain = np.asarray(
            [
                self.semantic_planar_task_error_gain[0],
                *([self.semantic_planar_task_error_gain[1]] * 3),
            ],
            dtype=np.float64,
        )
        correction_limit = np.asarray(
            [
                self.semantic_planar_task_correction_max_velocity[0],
                *(
                    [self.semantic_planar_task_correction_max_velocity[1]]
                    * 3
                ),
            ],
            dtype=np.float64,
        )
        correction_velocity = (
            np.zeros_like(task_error)
            if settle_hold
            else np.clip(
                error_gain * task_error,
                -correction_limit,
                correction_limit,
            )
        )
        feedforward_velocity = np.concatenate(
            (
                [task_velocity[0]],
                self.semantic_vertical_height_profile * task_velocity[1],
            )
        )
        desired_task_velocity = (
            np.zeros_like(feedforward_velocity)
            if settle_hold
            else feedforward_velocity + correction_velocity
        )
        desired = np.concatenate(
            (
                self.semantic_planar_task_weights * desired_task_velocity,
                np.zeros(3, dtype=np.float64),
            )
        )

        mobility = self.semantic_planar_joint_mobility.copy()
        follower_scale = (
            self.semantic_planar_follower_min_mobility_scale
            + (1.0 - self.semantic_planar_follower_min_mobility_scale)
            * float(np.clip(clearance_gate, 0.0, 1.0))
        )
        for local_index, joint_index in enumerate(columns):
            if int(joint_index) != self.semantic_vertical_primary_joint:
                mobility[local_index] *= follower_scale
        try:
            requested_joint_velocity = weighted_damped_least_squares(
                task,
                desired,
                self.semantic_vertical_damping,
                mobility,
            )
        except np.linalg.LinAlgError:
            self.logger.warn("B601 resolved-rate planar Jacobian is singular; braking")
            requested_joint_velocity = np.zeros(columns.size, dtype=np.float64)

        self.semantic_vertical_descent_allowed = bool(
            not self.semantic_vertical_descent_guard_enabled
            or task_velocity[1]
            < -self.semantic_vertical_descent_intent_deadband_m_s
        )
        guard_table_heights = (
            None
            if self.table_frame_heights is None
            else self.table_frame_heights[
                self._articulation_clearance_frame_ids
            ]
        )
        descent_guard_rows, descent_guard_mask = select_descent_guard_rows(
            height_rows,
            self._articulation_clearance_frame_ids,
            self._end_frame_id,
            current_heights,
            guard_table_heights,
            self.semantic_vertical_descent_guard_clearance_margin_m,
        )
        self.semantic_vertical_descent_guard_row_count = int(
            np.count_nonzero(descent_guard_mask)
        )
        requested_joint_velocity = suppress_uncommanded_task_descent(
            requested_joint_velocity,
            descent_guard_rows,
            self.semantic_vertical_descent_allowed,
        )
        joint_velocity_limit = self._semantic_outer_joint_velocity_limits()[columns]
        joint_velocity = coordinated_acceleration_limited_velocity(
            requested_joint_velocity,
            self.semantic_planar_joint_velocity_rad_s,
            joint_velocity_limit,
            self.semantic_planar_joint_max_acceleration,
            dt,
        )
        joint_velocity = suppress_uncommanded_task_descent(
            joint_velocity,
            descent_guard_rows,
            self.semantic_vertical_descent_allowed,
        )
        moving = np.abs(joint_velocity) > 1e-12
        if np.any(moving):
            velocity_scale = min(
                1.0,
                float(
                    np.min(
                        joint_velocity_limit[moving]
                        / np.abs(joint_velocity[moving])
                    )
                ),
            )
            joint_velocity *= velocity_scale
        q_request[columns] = previous_target[columns] + joint_velocity * dt
        lower, upper = self._semantic_joint_bounds(
            self.semantic_control_reference_q
        )
        q_request = np.clip(q_request, lower, upper)
        joint_velocity = (q_request[columns] - previous_target[columns]) / dt
        self.last_semantic_planar_error_m = np.asarray(
            [float(task_error[0]), float(np.max(np.abs(task_error[1:])))],
            dtype=np.float64,
        )
        self.last_semantic_vertical_error_m = self.last_semantic_planar_error_m[1]
        return q_request, joint_velocity, task_error

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
        lower, upper = self._semantic_joint_bounds(q_reference)
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

    def _project_semantic_target_to_table_clearance(
        self,
        current_q: np.ndarray,
        requested_q: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Project a semantic update onto the existing table-safe path.

        Rejecting an entire visual update at the table boundary leaves all
        smoothing state stale and can trap subsequent recovery commands.  A
        short bisection instead retains the largest verified fraction of the
        requested joint motion.  It uses exactly the same clearance predicate
        as the fail-closed guard; no table tolerance is relaxed.
        """

        current = np.asarray(current_q, dtype=np.float64).reshape(-1)
        requested = np.asarray(requested_q, dtype=np.float64).reshape(-1)
        if current.shape != requested.shape or not all(
            np.all(np.isfinite(value)) for value in (current, requested)
        ):
            raise ValueError("semantic clearance projection requires matching finite joints")
        if self._semantic_table_clearance_ok(current, requested):
            return requested.copy(), 1.0

        lower = 0.0
        upper = 1.0
        delta = requested - current
        for _ in range(12):
            fraction = 0.5 * (lower + upper)
            candidate = current + fraction * delta
            if self._semantic_table_clearance_ok(current, candidate):
                lower = fraction
            else:
                upper = fraction
        projected = current + lower * delta
        # Numerical noise in the final FK call must still fail closed.
        if not self._semantic_table_clearance_ok(current, projected):
            return current.copy(), 0.0
        return projected, lower

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
        # Bound one-frame vision innovations before they enter either the
        # motion detector or the low-pass filter.  This removes RTMPose jumps
        # without adding latency to ordinary continuous hand motion.
        raw = self.last_palm_raw_offset + np.clip(
            raw - self.last_palm_raw_offset,
            -self.palm_filter_max_innovation_m,
            self.palm_filter_max_innovation_m,
        )
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
        # One-Euro-style adaptive smoothing: stay quiet around a stationary
        # palm, but raise the bandwidth as soon as deliberate motion appears.
        speed_fraction = speed / (speed + self.palm_filter_speed_scale_m_s)
        fast_alpha = max(self.palm_filter_alpha, self.palm_filter_fast_alpha)
        alpha = float(
            np.clip(
                self.palm_filter_alpha
                + (fast_alpha - self.palm_filter_alpha) * speed_fraction,
                0.0,
                1.0,
            )
        )
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
