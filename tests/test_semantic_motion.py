import numpy as np
import pytest

from arm_control.semantic_motion import (
    build_operator_arm_motion,
    semantic_direct_joint_offsets,
    wrist_translation_anchor,
    wrap_angle_radians,
)


def test_build_operator_arm_motion_extracts_requested_human_channels():
    points_3d = {
        "pose_right_shoulder": np.array([-1.0, 0.0, 1.0]),
        "pose_right_elbow": np.array([0.0, 0.0, 1.0]),
        "pose_right_wrist": np.array([0.0, -1.0, 1.0]),
        "right_0": np.array([0.0, -1.0, 1.0]),
        "right_5": np.array([-0.1, -2.0, 1.0]),
        "right_9": np.array([0.0, -2.0, 1.0]),
        "right_17": np.array([0.1, -2.0, 1.0]),
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
    assert motion.elbow_angle_rad == pytest.approx(np.pi / 2.0)
    assert motion.wrist_pitch_rad == pytest.approx(0.0)
    assert motion.palm_roll_rad == pytest.approx(0.0)


def test_wrist_twist_tracks_palm_normal_around_forearm_axis():
    angle = np.deg2rad(30.0)
    points_3d = {
        "pose_right_shoulder": np.array([0.0, 0.0, 0.0]),
        "pose_right_elbow": np.array([1.0, 0.0, 0.0]),
        "pose_right_wrist": np.array([1.0, 1.0, 0.0]),
        "right_0": np.array([1.0, 1.0, 0.0]),
        "right_5": np.array([1.0 - 0.1 * np.cos(angle), 2.0, 0.1 * np.sin(angle)]),
        "right_9": np.array([1.0, 2.0, 0.0]),
        "right_17": np.array([1.0 + 0.1 * np.cos(angle), 2.0, -0.1 * np.sin(angle)]),
    }

    motion = build_operator_arm_motion(
        points_3d,
        {},
        "right",
        [0.0, 0.0, 0.0],
    )

    assert motion is not None
    assert motion.palm_roll_rad == pytest.approx(angle)


def test_wrist_twist_fails_closed_for_straight_arm_or_missing_depth():
    straight_arm = {
        "pose_right_shoulder": np.array([0.0, 0.0, 0.0]),
        "pose_right_elbow": np.array([0.0, 1.0, 0.0]),
        "right_0": np.array([0.0, 2.0, 0.0]),
        "right_5": np.array([-0.1, 3.0, 0.0]),
        "right_9": np.array([0.0, 3.0, 0.0]),
        "right_17": np.array([0.1, 3.0, 0.0]),
    }

    straight = build_operator_arm_motion(
        straight_arm, {}, "right", [0.0, 0.0, 0.0]
    )
    missing_depth = build_operator_arm_motion(
        {},
        {"right_5": [200.0, 100.0], "right_17": [100.0, 100.0]},
        "right",
        [0.0, 0.0, 0.0],
    )

    assert straight is not None and straight.palm_roll_rad is None
    assert missing_depth is not None and missing_depth.palm_roll_rad is None


def test_wrist_twist_uses_body_wrist_and_two_knuckle_depth_fallback():
    points_3d = {
        "pose_right_shoulder": np.array([0.0, 0.0, 0.0]),
        "pose_right_elbow": np.array([1.0, 0.0, 0.0]),
        "pose_right_wrist": np.array([1.0, 1.0, 0.0]),
        "right_5": np.array([0.9, 2.0, 0.0]),
        "right_17": np.array([1.1, 2.0, 0.0]),
    }

    motion = build_operator_arm_motion(
        points_3d, {}, "right", [0.0, 0.0, 0.0]
    )

    assert motion is not None
    assert motion.palm_roll_rad == pytest.approx(0.0)


def test_build_operator_arm_motion_fails_closed_without_control_hand():
    assert build_operator_arm_motion({}, {}, None, [0.0, 0.0, 0.0]) is None
    assert build_operator_arm_motion({}, {}, "right", None) is None


def test_wrist_translation_anchor_does_not_follow_rotating_knuckles():
    wrist = np.array([0.20, -0.10, 1.00])
    points = {
        "pose_right_wrist": wrist,
        "right_0": wrist + np.array([0.005, 0.0, 0.0]),
        # These points can move strongly as the wrist rotates; they must not
        # influence the translation anchor.
        "right_5": np.array([0.10, -0.25, 0.90]),
        "right_9": np.array([0.20, -0.35, 1.10]),
        "right_17": np.array([0.30, -0.20, 1.15]),
    }

    np.testing.assert_allclose(wrist_translation_anchor(points, "right"), wrist)


def test_wrist_translation_anchor_rejects_conflicting_depth_estimates():
    points = {
        "pose_right_wrist": np.array([0.0, 0.0, 1.0]),
        "right_0": np.array([0.0, 0.0, 1.25]),
    }

    assert wrist_translation_anchor(points, "right") is None


def test_semantic_direct_joint_mapping_uses_only_j1_and_j6():
    offsets = semantic_direct_joint_offsets(
        lateral_m=0.05,
        roll_delta_rad=0.30,
        gains=[0.6, -0.5, 0.2, 0.15, 0.2],
        channel_enable=[1, 1, 1, 1, 1, 0],
        max_joint_offset_rad=[0.1] * 6,
    )

    np.testing.assert_allclose(
        offsets,
        [0.03, 0.0, 0.0, 0.0, 0.0, 0.06],
    )


def test_semantic_joint_mapping_respects_channel_disable_and_limits():
    offsets = semantic_direct_joint_offsets(
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
