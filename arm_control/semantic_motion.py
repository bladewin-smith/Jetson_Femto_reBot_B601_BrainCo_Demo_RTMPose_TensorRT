"""Human-arm observations used by the B601 semantic joint controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OperatorArmMotion:
    """One camera-frame observation for semantic B601 joint control."""

    side: str
    palm_displacement_m: np.ndarray
    elbow_angle_rad: float | None
    wrist_pitch_rad: float | None
    palm_roll_rad: float | None


def wrap_angle_radians(value: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _point(
    points: Mapping[str, Sequence[float]], key: str, dimensions: int
) -> np.ndarray | None:
    value = points.get(key)
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size < dimensions or not np.all(np.isfinite(result[:dimensions])):
        return None
    return result[:dimensions]


def _joint_angle(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float | None:
    vector_a = first - middle
    vector_b = last - middle
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _elbow_angle(
    points_3d: Mapping[str, Sequence[float]],
    points_2d: Mapping[str, Sequence[float]],
    side: str,
) -> float | None:
    keys = (
        f"pose_{side}_shoulder",
        f"pose_{side}_elbow",
        f"pose_{side}_wrist",
    )
    points = [_point(points_3d, key, 3) for key in keys]
    if all(point is not None for point in points):
        return _joint_angle(points[0], points[1], points[2])

    points = [_point(points_2d, key, 2) for key in keys]
    if all(point is not None for point in points):
        return _joint_angle(points[0], points[1], points[2])
    return None


def _wrist_pitch(
    points_3d: Mapping[str, Sequence[float]], side: str
) -> float | None:
    """Return forearm-to-palm pitch in the camera vertical/depth plane."""

    elbow = _point(points_3d, f"pose_{side}_elbow", 3)
    wrist = _point(points_3d, f"{side}_0", 3)
    middle_mcp = _point(points_3d, f"{side}_9", 3)
    if elbow is None or wrist is None or middle_mcp is None:
        return None

    forearm = wrist - elbow
    palm_axis = middle_mcp - wrist
    # Camera Y points down and Z points away from the camera.  Projecting onto
    # (-Y, Z) isolates the small forward/back wrist flexion requested by the
    # semantic controller while ignoring lateral hand translation.
    forearm_projection = np.array([-forearm[1], forearm[2]], dtype=np.float64)
    palm_projection = np.array([-palm_axis[1], palm_axis[2]], dtype=np.float64)
    if min(
        float(np.linalg.norm(forearm_projection)),
        float(np.linalg.norm(palm_projection)),
    ) < 1e-5:
        return None
    forearm_angle = np.arctan2(forearm_projection[1], forearm_projection[0])
    palm_angle = np.arctan2(palm_projection[1], palm_projection[0])
    return wrap_angle_radians(palm_angle - forearm_angle)


def _palm_roll(
    points_2d: Mapping[str, Sequence[float]], side: str
) -> float | None:
    """Return palm rotation in the camera image plane."""

    index_mcp = _point(points_2d, f"{side}_5", 2)
    pinky_mcp = _point(points_2d, f"{side}_17", 2)
    if index_mcp is None or pinky_mcp is None:
        return None
    lateral = index_mcp - pinky_mcp
    if float(np.linalg.norm(lateral)) < 1e-5:
        return None
    return float(np.arctan2(-lateral[1], lateral[0]))


def build_operator_arm_motion(
    points_3d: Mapping[str, Sequence[float]],
    points_2d: Mapping[str, Sequence[float]],
    side: str | None,
    palm_displacement_m: Sequence[float] | None,
) -> OperatorArmMotion | None:
    """Build the semantic signals from RTMPose/Orbbec landmarks."""

    side = str(side or "").strip().lower()
    if side not in ("left", "right") or palm_displacement_m is None:
        return None
    displacement = np.asarray(palm_displacement_m, dtype=np.float64).reshape(-1)
    if displacement.size != 3 or not np.all(np.isfinite(displacement)):
        return None
    return OperatorArmMotion(
        side=side,
        palm_displacement_m=displacement.copy(),
        elbow_angle_rad=_elbow_angle(points_3d, points_2d, side),
        wrist_pitch_rad=_wrist_pitch(points_3d, side),
        palm_roll_rad=_palm_roll(points_2d, side),
    )


def semantic_direct_joint_offsets(
    lateral_m: float,
    roll_delta_rad: float,
    gains: Sequence[float],
    channel_enable: Sequence[float],
    max_joint_offset_rad: Sequence[float],
) -> np.ndarray:
    """Map the two direct human signals to bounded J1/J6 offsets.

    Palm depth and height are intentionally absent here.  They are solved as
    one Cartesian planar task over J2/J3/J4 by :class:`B601PalmFollower`.
    Human elbow and wrist angles are vision diagnostics only and no longer
    command individual B601 joints.
    """

    gains_array = np.asarray(gains, dtype=np.float64).reshape(-1)
    channels = np.asarray(channel_enable, dtype=np.float64).reshape(-1)
    maximum = np.asarray(max_joint_offset_rad, dtype=np.float64).reshape(-1)
    if gains_array.size != 5:
        raise ValueError("semantic gains must contain five values")
    if channels.size != 6:
        raise ValueError("semantic channel enable must contain six values")
    if maximum.size != 6 or np.any(maximum <= 0.0):
        raise ValueError("semantic maximum joint offsets must contain six positive values")

    offsets = np.zeros(6, dtype=np.float64)
    offsets[0] = channels[0] * gains_array[0] * float(lateral_m)
    offsets[5] = channels[4] * gains_array[4] * float(roll_delta_rad)
    return np.clip(offsets, -maximum, maximum)
