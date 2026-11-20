import numpy as np
import pytest

from arm_control.rebot_b601 import (
    B601PalmFollower,
    StartupGuardError,
    apply_tabletop_motion_envelope,
    joint_tracking_bounds,
    joint_target_safety_violations,
    limit_mit_target_by_effort,
    minimum_jerk_blend,
    ramp_joint_target_with_tracking_window,
    resolve_native_frame_ids,
    scale_vector_to_max_abs,
    send_mit_checked,
    synchronize_vertical_follower_progress,
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


def test_joint_limit_margin_allows_return_to_validated_tabletop_reference():
    lower = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    upper = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
    startup = [-1.5139, -0.0090, -0.0105, 0.0029, -0.0574, -0.3317]
    current = [-1.5158, -0.0586, -0.0650, 0.0425, -0.0532, -0.3317]
    target = [-1.5158, -0.0448, -0.0564, 0.0192, -0.0536, -0.3317]

    outside, jump = joint_target_safety_violations(
        target,
        current,
        lower,
        upper,
        0.05,
        [0.10, 0.10, 0.10, 0.14, 0.14, 0.14],
        startup,
    )

    assert not np.any(outside)
    assert not np.any(jump)


def test_joint_limit_margin_rejects_target_beyond_validated_tabletop_reference():
    lower = [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14]
    upper = [2.8, 0.0, 0.0, 1.57, 1.57, 3.14]
    startup = [-1.5139, -0.0090, -0.0105, 0.0029, -0.0574, -0.3317]
    current = [-1.5158, -0.0586, -0.0650, 0.0425, -0.0532, -0.3317]
    target = [-1.5158, -0.0050, -0.0564, 0.0192, -0.0536, -0.3317]

    outside, _ = joint_target_safety_violations(
        target,
        current,
        lower,
        upper,
        0.05,
        [0.10, 0.10, 0.10, 0.14, 0.14, 0.14],
        startup,
    )

    assert outside[1]


def test_joint_tracking_bounds_keep_validated_table_pose_reachable():
    lower, upper = joint_tracking_bounds(
        [-2.8, -3.14, -3.14],
        [2.8, 0.0, 0.0],
        0.05,
        [-1.5, -0.009, -0.011],
    )

    np.testing.assert_allclose(lower, [-2.75, -3.09, -3.09])
    np.testing.assert_allclose(upper, [2.75, -0.009, -0.011])


def test_coordinated_joint_step_scaling_preserves_direction():
    raw = np.array([-0.023, -0.044, 0.018])

    scaled = scale_vector_to_max_abs(raw, 0.012)

    assert np.max(np.abs(scaled)) == pytest.approx(0.012)
    np.testing.assert_allclose(scaled / raw, np.full(3, 0.012 / 0.044))


def test_pinocchio_frame_ids_are_converted_to_native_integers():
    class FakeModel:
        nframes = 10

        def getFrameId(self, name):
            return np.int64({"link3": 3, "link4": 4, "end_link": 9}[name])

    frame_ids = resolve_native_frame_ids(
        FakeModel(),
        ("link3", "link4", "end_link"),
    )

    assert frame_ids == [3, 4, 9]
    assert all(type(frame_id) is int for frame_id in frame_ids)


def test_semantic_target_can_accumulate_safe_lead_over_static_feedback():
    measured = np.zeros(3)
    target = measured.copy()
    desired = np.array([-0.05, -0.05, 0.03])

    for _ in range(20):
        target = ramp_joint_target_with_tracking_window(
            desired,
            measured,
            target,
            [0.002, 0.002, 0.0015],
            [0.012, 0.018, 0.010],
        )

    np.testing.assert_allclose(target, [-0.012, -0.018, 0.010])


def test_checked_mit_send_reports_joint_name_and_keeps_other_holds_alive():
    class FakeMotor:
        def __init__(self, failure=None):
            self.failure = failure
            self.commands = []

        def send_mit(self, *command):
            self.commands.append(command)
            if self.failure is not None:
                raise self.failure

    motors = {
        "joint1": FakeMotor(),
        "joint2": FakeMotor(RuntimeError("serial write rejected")),
        "joint3": FakeMotor(),
    }

    with pytest.raises(RuntimeError, match=r"joint2.*serial write rejected"):
        send_mit_checked(
            motors,
            ["joint1", "joint2", "joint3"],
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [120.0, 120.0, 120.0],
            [8.0, 8.0, 8.0],
            [1.0, 2.0, 3.0],
        )

    assert all(len(motor.commands) == 1 for motor in motors.values())
    assert motors["joint3"].commands[0] == (0.3, 0.0, 120.0, 8.0, 3.0)


def test_vertical_followers_cannot_outrun_measured_j3_progress():
    reference = np.array([0.0, -0.0105, -0.0105, -0.0036, 0.0, 0.0])
    desired = np.array([0.0, -0.0332, -0.0491, 0.0039, 0.0, 0.0])
    measured = np.array([0.0, -0.0315, -0.0219, 0.0113, 0.0, 0.0])

    target, primary_progress, follower_limit = synchronize_vertical_follower_progress(
        reference,
        desired,
        measured,
        desired,
        primary_index=2,
        follower_indices=[1, 3],
        max_follower_progress_lead=0.15,
    )

    expected_primary_progress = (-0.0219 + 0.0105) / (-0.0491 + 0.0105)
    assert primary_progress == pytest.approx(expected_primary_progress)
    assert follower_limit == pytest.approx(expected_primary_progress + 0.15)
    assert target[2] == pytest.approx(desired[2])
    assert target[1] == pytest.approx(
        reference[1] + follower_limit * (desired[1] - reference[1])
    )
    assert target[3] == pytest.approx(
        reference[3] + follower_limit * (desired[3] - reference[3])
    )


def test_table_clearance_allows_monotonic_recovery_from_small_tracking_dip():
    follower = object.__new__(B601PalmFollower)
    follower.table_frame_heights = np.array([0.0, 0.0])
    follower.semantic_table_clearance_tolerance_m = 0.001
    follower._frame_heights = lambda q: np.asarray(q, dtype=np.float64)

    assert follower._semantic_table_clearance_ok(
        np.array([-0.0015, 0.0]),
        np.array([0.0, 0.0]),
    )
    assert not follower._semantic_table_clearance_ok(
        np.array([-0.0015, 0.0]),
        np.array([-0.0020, 0.0]),
    )


def test_mit_target_is_clipped_by_total_effort_not_only_feedforward():
    target, effort, limited = limit_mit_target_by_effort(
        measured_position_rad=[0.0, 0.0],
        measured_velocity_rad_s=[0.0, 0.0],
        requested_target_rad=[1.0, -1.0],
        kp=[10.0, 10.0],
        kd=[1.0, 1.0],
        feedforward_torque_nm=[3.0, -3.0],
        effort_limit_nm=[5.0, 5.0],
    )

    np.testing.assert_allclose(target, [0.2, -0.2])
    np.testing.assert_allclose(effort, [5.0, -5.0])
    assert limited.tolist() == [True, True]


def test_stall_watchdog_accepts_slow_but_measurable_progress(monkeypatch):
    import arm_control.rebot_b601 as backend

    follower = object.__new__(B601PalmFollower)
    follower.control_mode = "semantic_joint"
    follower.semantic_control_reference_q = np.zeros(2)
    follower.semantic_stall_error = np.array([0.02, 0.02])
    follower.semantic_stall_min_progress_rad = 0.0015
    follower.semantic_stall_timeout_s = 3.0
    follower.semantic_stall_started_s = np.full(2, np.nan)
    follower.semantic_stall_reference_q = np.full(2, np.nan)
    follower.semantic_stall_direction = np.zeros(2)
    follower.semantic_actuation_fault = None

    times = iter([0.0, 2.0, 4.0])
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(times))
    target = np.array([0.05, 0.0])
    follower._update_semantic_stall_watchdog(
        np.array([0.0, 0.0]), np.zeros(2), target
    )
    follower._update_semantic_stall_watchdog(
        np.array([0.0016, 0.0]), np.zeros(2), target
    )
    follower._update_semantic_stall_watchdog(
        np.array([0.0032, 0.0]), np.zeros(2), target
    )

    assert follower.semantic_actuation_fault is None
