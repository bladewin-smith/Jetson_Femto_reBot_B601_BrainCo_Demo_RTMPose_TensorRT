import numpy as np
import pytest

from arm_control.rebot_b601 import (
    StartupGuardError,
    apply_tabletop_motion_envelope,
    joint_target_safety_violations,
    minimum_jerk_blend,
    validate_tabletop_startup,
)


def test_tabletop_startup_accepts_stationary_reference_pose():
    validate_tabletop_startup(
        [0.01, -0.02, 0.0, 0.03, 0.0, -0.01],
        [0.0] * 6,
        [0.0] * 6,
        [0.05] * 6,
        0.2,
    )


@pytest.mark.parametrize(
    "positions,velocities",
    [
        ([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 6),
        ([0.0] * 6, [0.0, 0.0, 0.0, 0.21, 0.0, 0.0]),
    ],
)
def test_tabletop_startup_rejects_wrong_or_moving_pose(positions, velocities):
    with pytest.raises(StartupGuardError):
        validate_tabletop_startup(positions, velocities, [0.0] * 6, 0.35, 0.2)


def test_tabletop_envelope_blocks_xy_until_lifted_and_never_moves_down():
    blocked = apply_tabletop_motion_envelope(
        [0.02, -0.02, -0.03], [0.03, 0.03, 0.0], [0.03, 0.03, 0.06], 0.02
    )
    np.testing.assert_allclose(blocked, [0.0, 0.0, 0.0])

    lifted = apply_tabletop_motion_envelope(
        [0.02, -0.02, 0.03], [0.03, 0.03, 0.0], [0.03, 0.03, 0.06], 0.02
    )
    np.testing.assert_allclose(lifted, [0.02, -0.02, 0.03])


def test_minimum_jerk_blend_has_expected_endpoints():
    assert minimum_jerk_blend(0.0) == pytest.approx(0.0)
    assert minimum_jerk_blend(1.0) == pytest.approx(1.0)
    assert minimum_jerk_blend(0.5) == pytest.approx(0.5)


def test_joint_limit_margin_allows_retreat_from_tabletop_upper_limit():
    lower = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    upper = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
    current = [-1.5139, -0.0090, -0.0105, 0.0029, -0.0574, -0.3317]
    target = [-1.5139, -0.0159, -0.0217, 0.0041, -0.0574, -0.3317]

    outside, jump = joint_target_safety_violations(
        target, current, lower, upper, 0.05, [0.10, 0.10, 0.10, 0.14, 0.14, 0.14]
    )

    assert not np.any(outside)
    assert not np.any(jump)


@pytest.mark.parametrize(
    "target",
    [
        [-1.5139, -0.0050, -0.0105, 0.0029, -0.0574, -0.3317],
        [-1.5139, 0.0010, -0.0105, 0.0029, -0.0574, -0.3317],
    ],
)
def test_joint_limit_margin_rejects_motion_toward_or_past_hard_stop(target):
    lower = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    upper = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
    current = [-1.5139, -0.0090, -0.0105, 0.0029, -0.0574, -0.3317]

    outside, _ = joint_target_safety_violations(
        target, current, lower, upper, 0.05, [0.10, 0.10, 0.10, 0.14, 0.14, 0.14]
    )

    assert outside[1]


def test_joint_limit_margin_rejects_entering_guard_zone_from_interior():
    lower = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    upper = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
    current = [-1.5139, -0.10, -0.10, 0.0029, -0.0574, -0.3317]
    target = [-1.5139, -0.03, -0.03, 0.0029, -0.0574, -0.3317]

    outside, _ = joint_target_safety_violations(
        target, current, lower, upper, 0.05, [0.10, 0.10, 0.10, 0.14, 0.14, 0.14]
    )

    assert outside[1]
    assert outside[2]
