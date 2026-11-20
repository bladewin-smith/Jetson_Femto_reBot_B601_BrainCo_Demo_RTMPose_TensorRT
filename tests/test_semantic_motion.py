import numpy as np
import pytest

from arm_control.semantic_motion import (
    build_operator_arm_motion,
    semantic_joint_offsets,
    wrap_angle_radians,
)


def test_build_operator_arm_motion_extracts_requested_human_channels():
    points_3d = {
        "pose_right_shoulder": np.array([0.0, 1.0, 1.0]),
        "pose_right_elbow": np.array([0.0, 0.0, 1.0]),
        "pose_right_wrist": np.array([0.0, -1.0, 1.0]),
        "right_0": np.array([0.0, -1.0, 1.0]),
        "right_9": np.array([0.0, -1.0, 2.0]),
    }
    points_2d = {
        "right_5": np.array([200.0, 100.0]),
        "right_17": np.array([100.0, 100.0]),
    }

    motion = build_operator_arm_motion(
        points_3d,
        points_2d,
        "right",
        [0.03, -0.02, -0.04],
    )

    assert motion is not None
    np.testing.assert_allclose(motion.palm_displacement_m, [0.03, -0.02, -0.04])
    assert motion.elbow_angle_rad == pytest.approx(np.pi)
    assert motion.wrist_pitch_rad == pytest.approx(np.pi / 2.0)
    assert motion.palm_roll_rad == pytest.approx(0.0)


def test_build_operator_arm_motion_fails_closed_without_control_hand():
    assert build_operator_arm_motion({}, {}, None, [0.0, 0.0, 0.0]) is None
    assert build_operator_arm_motion({}, {}, "right", None) is None


def test_semantic_joint_mapping_uses_j1_j2_j3_j4_and_j6_only():
    offsets = semantic_joint_offsets(
        lateral_m=0.05,
        forward_m=0.04,
        elbow_delta_rad=0.20,
        wrist_delta_rad=-0.10,
        roll_delta_rad=0.30,
        gains=[0.6, -0.5, 0.2, 0.15, 0.2],
        channel_enable=[1, 1, 1, 1, 1, 0],
        max_joint_offset_rad=[0.1] * 6,
    )

    np.testing.assert_allclose(
        offsets,
        [0.03, -0.02, 0.04, -0.015, 0.0, 0.06],
    )


def test_semantic_joint_mapping_respects_channel_disable_and_limits():
    offsets = semantic_joint_offsets(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        gains=[1.0] * 5,
        channel_enable=[1, 0, 0, 0, 0, 0],
        max_joint_offset_rad=[0.02] * 6,
    )

    np.testing.assert_allclose(offsets, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "value,expected",
    [
        (3.0 * np.pi, -np.pi),
        (-3.0 * np.pi, -np.pi),
        (0.25, 0.25),
    ],
)
def test_wrap_angle_radians(value, expected):
    assert wrap_angle_radians(value) == pytest.approx(expected)
