import numpy as np
import pytest

from arm_control.rebot_b601 import (
    StartupGuardError,
    apply_tabletop_motion_envelope,
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
