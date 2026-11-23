import numpy as np
import pytest

from arm_control.rebot_b601 import (
    B601PalmFollower,
    StartupGuardError,
    acceleration_limited_scalar_target,
    acceleration_limited_scalar_target_with_tracking_window,
    acceleration_limited_vector_target,
    adaptive_goal_velocity_hold_s,
    adaptive_position_correction_fraction,
    apply_tabletop_motion_envelope,
    bounded_wall_clock_trajectory_dt,
    coupled_stream_velocity_limits,
    confirmed_stationary_planar_target,
    confirm_vector_reversals,
    confirm_directional_wrist_twist,
    coordinated_acceleration_limited_velocity,
    filtered_goal_velocity,
    filtered_wrapped_angle_delta,
    gate_outward_scalar_increment,
    horizontal_radial_axis,
    independent_planar_task_backpressure,
    intersect_joint_position_limits,
    jerk_limited_joint_target,
    joint_tracking_bounds,
    joint_target_safety_violations,
    limit_mit_target_by_effort,
    minimum_jerk_blend,
    ramp_joint_target_with_tracking_window,
    rate_limited_unit_interval,
    resolve_native_frame_ids,
    rotation_compensated_scalar_observation,
    scale_vector_to_max_abs,
    select_descent_guard_rows,
    send_mit_checked,
    slew_limit_vector,
    smoothed_semantic_goal_velocity,
    synchronize_vertical_follower_progress,
    suppress_uncommanded_task_descent,
    task_space_backpressure,
    validate_wrapped_angle_observation,
    validate_tabletop_startup,
    weighted_damped_least_squares,
    wrist_height_release_ready,
    wrist_hold_vertical_planner_state,
    wrapped_rate_motion_hold,
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


def test_smooth_follow_safety_delta_contains_every_semantic_tracking_window():
    current = np.zeros(6)
    target = np.asarray([0.032, -0.055, -0.110, 0.060, 0.008, 0.090])
    safety_delta = [0.045, 0.065, 0.125, 0.075, 0.012, 0.105]

    outside, jump = joint_target_safety_violations(
        target,
        current,
        [-3.0] * 6,
        [3.0] * 6,
        0.10,
        safety_delta,
    )

    assert not np.any(outside)
    assert not np.any(jump)
    assert np.all(np.asarray(safety_delta) > np.abs(target))


def test_old_j3_safety_delta_rejects_the_new_tracking_window():
    _, jump = joint_target_safety_violations(
        [0.0, 0.0, -0.110, 0.0, 0.0, 0.0],
        [0.0] * 6,
        [-3.0] * 6,
        [3.0] * 6,
        0.10,
        [0.10, 0.10, 0.10, 0.14, 0.14, 0.14],
    )

    assert jump[2]


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


def test_official_safety_envelope_uses_conservative_urdf_intersection():
    lower, upper = intersect_joint_position_limits(
        [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14],
        [2.8, 0.0, 0.0, 1.57, 1.57, 3.14],
        [-2.6005, -3.5954, -3.5954, -1.4992, -1.4992, -1.4992],
        [2.6005, 0.0, 0.0, 1.4992, 1.4992, 1.4992],
    )

    np.testing.assert_allclose(
        lower, [-2.6005, -3.14, -3.14, -1.4992, -1.4992, -1.4992]
    )
    np.testing.assert_allclose(
        upper, [2.6005, 0.0, 0.0, 1.4992, 1.4992, 1.4992]
    )


def test_official_safety_envelope_rejects_non_overlapping_ranges():
    with pytest.raises(ValueError, match="do not overlap"):
        intersect_joint_position_limits([-1.0], [0.0], [0.1], [1.0])


def test_official_envelope_retains_margin_without_trapping_table_pose():
    lower, upper = joint_tracking_bounds(
        [-2.6005, -3.14, -3.14, -1.4992, -1.4992, -1.4992],
        [2.6005, 0.0, 0.0, 1.4992, 1.4992, 1.4992],
        0.10,
        [-1.6173, -0.0105, -0.0498, 0.1383, -0.0116, -0.4606],
    )

    np.testing.assert_allclose(
        lower, [-2.5005, -3.04, -3.04, -1.3992, -1.3992, -1.3992]
    )
    np.testing.assert_allclose(
        upper, [2.5005, -0.0105, -0.0498, 1.3992, 1.3992, 1.3992]
    )


def test_coordinated_joint_step_scaling_preserves_direction():
    raw = np.array([-0.023, -0.044, 0.018])

    scaled = scale_vector_to_max_abs(raw, 0.012)

    assert np.max(np.abs(scaled)) == pytest.approx(0.012)
    np.testing.assert_allclose(scaled / raw, np.full(3, 0.012 / 0.044))


def test_semantic_joint_ramp_preserves_coordinated_planar_direction():
    target = ramp_joint_target_with_tracking_window(
        desired_rad=[0.0, 0.20, -0.40, 0.10, 0.0, 0.0],
        measured_rad=[0.0] * 6,
        previous_target_rad=[0.0] * 6,
        max_step_rad=[0.01, 0.02, 0.03, 0.01, 0.01, 0.01],
        max_tracking_error_rad=[0.50] * 6,
        coordinated_indices=(1, 2, 3),
    )

    np.testing.assert_allclose(target[1:4], [0.015, -0.03, 0.0075])
    np.testing.assert_allclose(
        target[1:4] / np.array([0.20, -0.40, 0.10]),
        np.full(3, 0.075),
    )


def test_planar_command_slew_limit_rejects_single_frame_jump():
    limited = slew_limit_vector(
        requested=[0.035, 0.025],
        previous=[0.0, 0.0],
        max_velocity=[0.040, 0.035],
        dt_s=0.06,
    )

    np.testing.assert_allclose(limited, [0.0024, 0.0021])


def test_planar_acceleration_limiter_brakes_before_reversing():
    target = np.zeros(2)
    velocity = np.zeros(2)
    target, velocity = acceleration_limited_vector_target(
        requested=[0.04, 0.03],
        previous=target,
        previous_velocity=velocity,
        max_velocity=[0.05, 0.04],
        max_acceleration=[0.20, 0.16],
        deadband=[0.0001, 0.0001],
        dt_s=0.05,
    )
    np.testing.assert_allclose(velocity, [0.01, 0.008])

    _, reversed_velocity = acceleration_limited_vector_target(
        requested=[-0.04, -0.03],
        previous=target,
        previous_velocity=velocity,
        max_velocity=[0.05, 0.04],
        max_acceleration=[0.20, 0.16],
        deadband=[0.0001, 0.0001],
        dt_s=0.05,
    )
    np.testing.assert_allclose(reversed_velocity, [0.0, 0.0], atol=1e-12)


def test_motion_reversal_requires_consecutive_samples_and_ignores_jitter():
    stable = np.zeros(3)
    active = np.zeros(3, dtype=np.int8)
    candidate = np.zeros(3, dtype=np.int8)
    count = np.zeros(3, dtype=np.int64)

    stable, active, candidate, count = confirm_vector_reversals(
        [-0.05, 0.03, 0.04],
        stable,
        active,
        candidate,
        count,
        [0.006, 0.006, 0.005],
        4,
    )
    np.testing.assert_allclose(stable, [-0.05, 0.03, 0.04])
    np.testing.assert_array_equal(active, [-1, 1, 1])

    # Same-direction changes smaller than the reversal deadband remain
    # continuous; the deadband applies only to a direction flip.
    stable, active, candidate, count = confirm_vector_reversals(
        [-0.052, 0.032, 0.042],
        stable,
        active,
        candidate,
        count,
        [0.006, 0.006, 0.005],
        4,
    )
    np.testing.assert_allclose(stable, [-0.052, 0.032, 0.042])
    np.testing.assert_array_equal(candidate, [0, 0, 0])

    # One large contradictory depth frame and one small lateral wobble are held.
    stable, active, candidate, count = confirm_vector_reversals(
        [0.02, -0.04, 0.038],
        stable,
        active,
        candidate,
        count,
        [0.006, 0.006, 0.005],
        4,
    )
    np.testing.assert_allclose(stable, [-0.052, 0.032, 0.042])
    np.testing.assert_array_equal(candidate, [1, -1, 0])
    np.testing.assert_array_equal(count, [1, 1, 0])

    for _ in range(3):
        stable, active, candidate, count = confirm_vector_reversals(
            [0.02, -0.04, 0.038],
            stable,
            active,
            candidate,
            count,
            [0.006, 0.006, 0.005],
            4,
        )
    np.testing.assert_allclose(stable, [0.02, -0.04, 0.042])
    np.testing.assert_array_equal(active, [1, -1, 1])


def test_j4_target_limiter_bounds_velocity_acceleration_and_deadband():
    target = 0.0
    velocity = 0.0
    velocities = []
    for _ in range(8):
        target, velocity = acceleration_limited_scalar_target(
            requested=0.08,
            previous=target,
            previous_velocity=velocity,
            max_velocity=0.025,
            max_acceleration=0.10,
            deadband=0.0015,
            dt_s=0.06,
        )
        velocities.append(velocity)

    assert 0.0 < target < 0.08
    assert max(abs(value) for value in velocities) <= 0.025
    assert max(abs(b - a) for a, b in zip(velocities, velocities[1:])) <= 0.006001

    _, reversed_velocity = acceleration_limited_scalar_target(
        requested=-0.08,
        previous=target,
        previous_velocity=velocity,
        max_velocity=0.025,
        max_acceleration=0.10,
        deadband=0.0015,
        dt_s=0.06,
    )
    assert reversed_velocity == pytest.approx(velocity - 0.006)

    arrived, stopped = acceleration_limited_scalar_target(
        requested=target + 0.001,
        previous=target,
        previous_velocity=velocity,
        max_velocity=0.025,
        max_acceleration=0.10,
        deadband=0.0015,
        dt_s=0.06,
    )
    assert arrived == pytest.approx(target + 0.001)
    assert stopped == pytest.approx(0.0)


def test_direct_joint_target_accelerates_before_tracking_window_is_applied():
    first_target, first_velocity = (
        acceleration_limited_scalar_target_with_tracking_window(
            requested=0.50,
            measured=0.0,
            previous=0.0,
            previous_velocity=0.0,
            max_velocity=0.10,
            max_acceleration=0.20,
            deadband=0.001,
            max_tracking_error=0.05,
            dt_s=0.10,
        )
    )
    second_target, second_velocity = (
        acceleration_limited_scalar_target_with_tracking_window(
            requested=0.50,
            measured=0.0,
            previous=first_target,
            previous_velocity=first_velocity,
            max_velocity=0.10,
            max_acceleration=0.20,
            deadband=0.001,
            max_tracking_error=0.05,
            dt_s=0.10,
        )
    )

    assert first_velocity == pytest.approx(0.02)
    assert second_velocity == pytest.approx(0.04)
    assert first_target == pytest.approx(0.002)
    assert second_target == pytest.approx(0.006)


def test_high_rate_joint_stream_is_continuous_and_bounded():
    position = np.zeros(2)
    velocity = np.zeros(2)
    acceleration = np.zeros(2)
    positions = []
    accelerations = []
    dt = 0.002
    for _ in range(800):
        position, velocity, acceleration = jerk_limited_joint_target(
            requested=[0.10, -0.06],
            previous=position,
            previous_velocity=velocity,
            previous_acceleration=acceleration,
            max_velocity=[0.20, 0.15],
            max_acceleration=[1.0, 0.8],
            max_jerk=[20.0, 16.0],
            dt_s=dt,
        )
        positions.append(position.copy())
        accelerations.append(acceleration.copy())

    positions = np.asarray(positions)
    accelerations = np.asarray(accelerations)
    assert np.all(np.diff(positions[:, 0]) >= -1e-12)
    assert np.all(np.diff(positions[:, 1]) <= 1e-12)
    assert np.max(np.abs(np.diff(accelerations, axis=0) / dt), axis=0)[0] <= 20.0001
    assert np.max(np.abs(np.diff(accelerations, axis=0) / dt), axis=0)[1] <= 16.0001
    np.testing.assert_allclose(position, [0.10, -0.06], atol=1e-6)


def test_high_rate_joint_stream_holds_position_and_clears_reversal_momentum():
    position, velocity, acceleration = jerk_limited_joint_target(
        requested=[-0.10],
        previous=[0.02],
        previous_velocity=[0.10],
        previous_acceleration=[0.0],
        max_velocity=[0.20],
        max_acceleration=[1.0],
        max_jerk=[20.0],
        dt_s=0.002,
    )

    assert position[0] == pytest.approx(0.02)
    assert velocity[0] == pytest.approx(0.0)
    assert acceleration[0] == pytest.approx(0.0)


def test_stream_velocity_caps_follow_the_outer_semantic_rate():
    effective = coupled_stream_velocity_limits(
        configured_limit=[0.25, 0.30, 0.35, 0.15, 0.10, 0.25],
        semantic_max_step=[0.0036, 0.0044, 0.0056, 0.0024, 0.0008, 0.0064],
        command_interval_s=0.04,
        outer_velocity_limit=[0.10, 10.0, 10.0, 0.045, 10.0, 0.10],
        follow_scale=1.35,
    )

    np.testing.assert_allclose(
        effective,
        [0.1215, 0.1485, 0.1890, 0.06075, 0.0270, 0.1350],
    )


def test_stream_velocity_caps_accept_current_planar_follow_scale():
    effective = coupled_stream_velocity_limits(
        configured_limit=[0.25, 0.30, 0.35, 0.15, 0.10, 0.30],
        semantic_max_step=[0.0036, 0.0044, 0.0056, 0.0024, 0.0008, 0.0072],
        command_interval_s=0.04,
        outer_velocity_limit=[0.10, 10.0, 10.0, 0.045, 10.0, 0.16],
        follow_scale=[1.30, 1.15, 1.15, 1.15, 1.30, 1.30],
    )

    np.testing.assert_allclose(
        effective,
        [0.1170, 0.1265, 0.1610, 0.05175, 0.0260, 0.2080],
    )


def test_adaptive_position_correction_is_quiet_near_goal():
    fraction, lead_ratio = adaptive_position_correction_fraction(
        tracking_error_rad=[0.0055, 0.0110, 0.0060],
        tracking_limit_rad=[0.055, 0.110, 0.060],
        minimum_fraction=0.18,
        maximum_fraction=0.32,
    )

    assert lead_ratio == pytest.approx(0.10)
    assert fraction == pytest.approx(0.18)


def test_adaptive_position_correction_reaches_full_catch_up_at_window_edge():
    fraction, lead_ratio = adaptive_position_correction_fraction(
        tracking_error_rad=[0.0, 0.099, 0.0],
        tracking_limit_rad=[0.055, 0.110, 0.060],
        minimum_fraction=0.18,
        maximum_fraction=0.32,
    )

    assert lead_ratio == pytest.approx(0.90)
    assert fraction == pytest.approx(0.32)


def test_adaptive_position_correction_blends_smoothly_mid_window():
    fraction, lead_ratio = adaptive_position_correction_fraction(
        tracking_error_rad=[0.0, 0.07425, 0.0],
        tracking_limit_rad=[0.055, 0.110, 0.060],
        minimum_fraction=0.18,
        maximum_fraction=0.32,
    )

    assert lead_ratio == pytest.approx(0.675)
    assert fraction == pytest.approx(0.25)


def test_goal_velocity_hold_adapts_to_slow_vision_without_exceeding_cap():
    assert adaptive_goal_velocity_hold_s(0.04, 0.04, 2.0, 0.20) == pytest.approx(
        0.08
    )
    assert adaptive_goal_velocity_hold_s(0.04, 0.10, 2.0, 0.20) == pytest.approx(
        0.15
    )
    assert adaptive_goal_velocity_hold_s(0.04, 0.30, 2.0, 0.20) == pytest.approx(
        0.20
    )
    assert adaptive_goal_velocity_hold_s(0.04, 0.30, 2.0, 0.55) == pytest.approx(
        0.45
    )


def test_wall_clock_trajectory_dt_uses_real_controller_period():
    assert bounded_wall_clock_trajectory_dt(0.0042, 0.002, 0.012) == pytest.approx(
        0.0042
    )


def test_wall_clock_trajectory_dt_caps_stalls_and_recovers_invalid_samples():
    assert bounded_wall_clock_trajectory_dt(0.050, 0.002, 0.012) == pytest.approx(
        0.012
    )
    assert bounded_wall_clock_trajectory_dt(float("nan"), 0.002, 0.012) == (
        pytest.approx(0.002)
    )
    with pytest.raises(ValueError):
        bounded_wall_clock_trajectory_dt(0.004, 0.002, 0.001)


def test_goal_velocity_preserves_planar_group_shape_and_resets_on_reversal():
    previous = np.zeros(6)
    requested = np.asarray([0.0, 0.0044, -0.0056, 0.0024, 0.0, 0.0])
    limits = np.asarray([0.10, 0.11, 0.14, 0.045, 0.02, 0.10])
    velocity = filtered_goal_velocity(
        requested,
        previous,
        np.zeros(6),
        limits,
        filter_alpha=1.0,
        dt_s=0.04,
        coordinated_indices=(1, 2, 3),
    )

    np.testing.assert_allclose(velocity[1:4], [0.0825, -0.105, 0.045])
    reversed_velocity = filtered_goal_velocity(
        -requested,
        previous,
        velocity,
        limits,
        filter_alpha=0.5,
        dt_s=0.04,
        coordinated_indices=(1, 2, 3),
    )
    np.testing.assert_array_equal(np.sign(reversed_velocity[1:4]), [-1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        np.abs(reversed_velocity[1:4]),
        0.5 * np.abs(velocity[1:4]),
    )


def test_semantic_goal_velocity_filters_planar_group_but_keeps_direct_rates():
    velocity = smoothed_semantic_goal_velocity(
        requested_position=[0.0, 0.0044, -0.0056, 0.0024, 0.0, 0.0],
        previous_position=[0.0] * 6,
        previous_velocity=[0.0] * 6,
        max_velocity=[0.10, 0.11, 0.14, 0.045, 0.02, 0.16],
        filter_alpha=0.45,
        dt_s=0.04,
        j1_velocity_rad_s=0.07,
        j6_velocity_rad_s=-0.12,
    )

    np.testing.assert_allclose(
        velocity[1:4], 0.45 * np.asarray([0.0825, -0.105, 0.045])
    )
    assert velocity[0] == pytest.approx(0.07)
    assert velocity[5] == pytest.approx(-0.12)


def test_high_rate_stream_preloads_velocity_without_overshooting_latest_goal():
    position = np.zeros(1)
    velocity = np.zeros(1)
    acceleration = np.zeros(1)
    for _ in range(50):
        position, velocity, acceleration = jerk_limited_joint_target(
            requested=[0.0],
            previous=position,
            previous_velocity=velocity,
            previous_acceleration=acceleration,
            max_velocity=[0.20],
            max_acceleration=[1.0],
            max_jerk=[20.0],
            dt_s=0.002,
            requested_velocity=[0.10],
        )

    assert position[0] == pytest.approx(0.0)
    assert velocity[0] > 0.0
    warm_position, _, _ = jerk_limited_joint_target(
        requested=[0.002],
        previous=position,
        previous_velocity=velocity,
        previous_acceleration=acceleration,
        max_velocity=[0.20],
        max_acceleration=[1.0],
        max_jerk=[20.0],
        dt_s=0.002,
        requested_velocity=[0.10],
    )
    cold_position, _, _ = jerk_limited_joint_target(
        requested=[0.002],
        previous=[0.0],
        previous_velocity=[0.0],
        previous_acceleration=[0.0],
        max_velocity=[0.20],
        max_acceleration=[1.0],
        max_jerk=[20.0],
        dt_s=0.002,
        requested_velocity=[0.0],
    )
    assert cold_position[0] < warm_position[0] <= 0.002


def test_clearance_gate_has_hysteresis_and_asymmetric_rate_limits():
    rising = rate_limited_unit_interval(
        requested=1.0,
        previous=0.0,
        rise_rate_s=1.5,
        fall_rate_s=3.0,
        dt_s=0.04,
        deadband=0.03,
    )
    held = rate_limited_unit_interval(
        requested=rising + 0.02,
        previous=rising,
        rise_rate_s=1.5,
        fall_rate_s=3.0,
        dt_s=0.04,
        deadband=0.03,
    )
    falling = rate_limited_unit_interval(
        requested=0.0,
        previous=0.60,
        rise_rate_s=1.5,
        fall_rate_s=3.0,
        dt_s=0.04,
        deadband=0.03,
    )

    assert rising == pytest.approx(0.06)
    assert held == pytest.approx(rising)
    assert falling == pytest.approx(0.48)


def test_clearance_gate_never_creates_motion_opposite_to_operator_target():
    assert gate_outward_scalar_increment(0.050, 0.020, 0.25) == pytest.approx(
        0.0275
    )
    # A falling clearance gate does not shrink an already accepted command.
    assert gate_outward_scalar_increment(0.050, 0.050, 0.0) == pytest.approx(
        0.050
    )
    # Returning toward zero and reversing direction are never blocked.
    assert gate_outward_scalar_increment(0.010, 0.050, 0.0) == pytest.approx(
        0.010
    )
    assert gate_outward_scalar_increment(-0.010, 0.050, 0.0) == pytest.approx(
        -0.010
    )


def test_task_backpressure_blocks_backlog_growth_but_allows_recovery():
    mid = task_space_backpressure(
        signed_error=[0.015, 0.0],
        requested_increment=[0.002, 0.0],
        max_tracking_error=[0.020, 0.020],
        start_fraction=0.50,
    )
    blocked = task_space_backpressure(
        signed_error=[0.025, 0.0],
        requested_increment=[0.002, 0.0],
        max_tracking_error=[0.020, 0.020],
        start_fraction=0.50,
    )
    recovery = task_space_backpressure(
        signed_error=[0.025, 0.0],
        requested_increment=[-0.002, 0.0],
        max_tracking_error=[0.020, 0.020],
        start_fraction=0.50,
    )

    assert mid == pytest.approx(0.5)
    assert blocked == pytest.approx(0.0)
    assert recovery == pytest.approx(1.0)


def test_planar_backpressure_does_not_cross_couple_forward_and_lift():
    forward_blocked = independent_planar_task_backpressure(
        signed_task_error=[0.025, 0.0, 0.0, 0.0],
        requested_planar_increment=[0.002, 0.003],
        vertical_height_profile=[0.5, 0.8, 1.0],
        max_tracking_error=[0.020, 0.020, 0.020, 0.020],
        start_fraction=0.50,
    )
    vertical_blocked = independent_planar_task_backpressure(
        signed_task_error=[0.0, 0.025, 0.025, 0.025],
        requested_planar_increment=[0.002, 0.003],
        vertical_height_profile=[0.5, 0.8, 1.0],
        max_tracking_error=[0.020, 0.020, 0.020, 0.020],
        start_fraction=0.50,
    )

    np.testing.assert_allclose(forward_blocked, [0.0, 1.0])
    np.testing.assert_allclose(vertical_blocked, [1.0, 0.0])


def test_coordinated_velocity_limiter_preserves_joint_ratio():
    first = coordinated_acceleration_limited_velocity(
        requested_velocity=[0.10, 0.20, -0.10],
        previous_velocity=[0.0, 0.0, 0.0],
        max_velocity=[0.10, 0.20, 0.10],
        max_acceleration=[0.50, 1.00, 0.50],
        dt_s=0.10,
    )
    second = coordinated_acceleration_limited_velocity(
        requested_velocity=[0.10, 0.20, -0.10],
        previous_velocity=first,
        max_velocity=[0.10, 0.20, 0.10],
        max_acceleration=[0.50, 1.00, 0.50],
        dt_s=0.10,
    )

    np.testing.assert_allclose(first, [0.05, 0.10, -0.05])
    np.testing.assert_allclose(second, [0.10, 0.20, -0.10])


def test_stream_position_correction_substantially_limits_chasing_speed():
    position = np.zeros(1)
    velocity = np.zeros(1)
    acceleration = np.zeros(1)
    observed_velocity = []
    for _ in range(500):
        position, velocity, acceleration = jerk_limited_joint_target(
            requested=[0.10],
            previous=position,
            previous_velocity=velocity,
            previous_acceleration=acceleration,
            max_velocity=[0.20],
            max_acceleration=[1.0],
            max_jerk=[20.0],
            dt_s=0.002,
            requested_velocity=[0.0],
            position_correction_gain_s=5.0,
            position_correction_max_velocity=[0.04],
        )
        observed_velocity.append(abs(float(velocity[0])))

    # Jerk-limited braking can briefly carry internal velocity past the 0.04
    # correction request, but it must remain far below the 0.20 hard cap.
    assert max(observed_velocity) < 0.07
    assert 0.0 < position[0] < 0.10


def test_stream_reversal_clears_stale_internal_momentum_without_position_step():
    position, velocity, acceleration = jerk_limited_joint_target(
        requested=[-0.10],
        previous=[0.0],
        previous_velocity=[0.08],
        previous_acceleration=[0.20],
        max_velocity=[0.20],
        max_acceleration=[1.0],
        max_jerk=[20.0],
        dt_s=0.01,
        requested_velocity=[-0.05],
        position_correction_gain_s=5.0,
        position_correction_max_velocity=[0.04],
    )

    np.testing.assert_allclose(position, [0.0])
    np.testing.assert_allclose(velocity, [0.0])
    np.testing.assert_allclose(acceleration, [0.0])


def test_wrist_twist_filter_wraps_boundary_and_rejects_one_frame_spike():
    boundary_delta = filtered_wrapped_angle_delta(
        value=np.deg2rad(-179.0),
        reference=np.deg2rad(179.0),
        previous_delta=0.0,
        deadband_rad=0.0,
        max_innovation_rad=np.deg2rad(20.0),
        filter_alpha=1.0,
    )
    spike_delta = filtered_wrapped_angle_delta(
        value=np.deg2rad(155.0),
        reference=0.0,
        previous_delta=0.0,
        deadband_rad=0.0,
        max_innovation_rad=np.deg2rad(20.0),
        filter_alpha=0.30,
    )

    assert np.rad2deg(boundary_delta) == pytest.approx(2.0)
    assert np.rad2deg(spike_delta) == pytest.approx(6.0)


def test_raw_wrist_observation_gate_wraps_boundary_and_rejects_depth_flip():
    boundary_ok, boundary_innovation = validate_wrapped_angle_observation(
        value=np.deg2rad(-179.0),
        previous_value=np.deg2rad(179.0),
        dt_s=0.10,
        max_rate_rad_s=np.deg2rad(180.0),
        max_jump_rad=np.deg2rad(25.0),
    )
    spike_ok, spike_innovation = validate_wrapped_angle_observation(
        value=np.deg2rad(-122.0),
        previous_value=np.deg2rad(20.0),
        dt_s=0.133,
        max_rate_rad_s=np.deg2rad(180.0),
        max_jump_rad=np.deg2rad(25.0),
    )

    assert boundary_ok
    assert np.rad2deg(boundary_innovation) == pytest.approx(2.0)
    assert not spike_ok
    assert abs(np.rad2deg(spike_innovation)) > 100.0


def test_wrist_rotation_hold_rebases_height_without_delayed_drop():
    held, bias = rotation_compensated_scalar_observation(
        raw_value=0.025,
        previous_intent=0.060,
        observation_bias=0.0,
        hold_active=True,
        lower_bound=0.0,
        upper_bound=0.060,
    )
    released, retained_bias = rotation_compensated_scalar_observation(
        raw_value=0.025,
        previous_intent=held,
        observation_bias=bias,
        hold_active=False,
        lower_bound=0.0,
        upper_bound=0.060,
    )
    deliberate_down, _ = rotation_compensated_scalar_observation(
        raw_value=0.005,
        previous_intent=released,
        observation_bias=retained_bias,
        hold_active=False,
        lower_bound=0.0,
        upper_bound=0.060,
    )

    assert held == pytest.approx(0.060)
    assert bias == pytest.approx(0.035)
    assert released == pytest.approx(0.060)
    assert deliberate_down == pytest.approx(0.040)


def test_logged_wrist_rotation_trace_cannot_create_vertical_drop():
    # Reduced trace from the hardware log: visual lift fell 60 -> 25 mm while
    # the filtered wrist angle increased.  The final invalid-depth sample must
    # also keep the last accepted height.
    raw_lifts = [0.060, 0.0588, 0.0488, 0.0429, 0.0331, 0.0250, 0.0]
    wrist_deltas_deg = [9.8, 18.6, 24.2, 29.0, 44.1, 48.7, 0.0]
    validity = [True, True, True, True, True, True, False]
    hold = False
    release_count = 0
    previous_delta = np.deg2rad(5.0)
    intent = 0.060
    bias = 0.0
    observed_intents = []

    for raw_lift, delta_deg, valid in zip(
        raw_lifts, wrist_deltas_deg, validity
    ):
        delta = np.deg2rad(delta_deg)
        hold, release_count, _ = wrapped_rate_motion_hold(
            current_delta_rad=delta,
            previous_delta_rad=previous_delta,
            dt_s=1.0,
            hold_active=hold,
            release_count=release_count,
            enter_rate_rad_s=np.deg2rad(3.0),
            release_rate_rad_s=np.deg2rad(1.0),
            release_samples=4,
            observation_valid=valid,
        )
        intent, bias = rotation_compensated_scalar_observation(
            raw_value=raw_lift,
            previous_intent=intent,
            observation_bias=bias,
            hold_active=hold,
            lower_bound=0.0,
            upper_bound=0.060,
        )
        observed_intents.append(intent)
        previous_delta = delta

    np.testing.assert_allclose(observed_intents, [0.060] * len(raw_lifts))


def test_wrist_motion_hold_requires_stable_release_samples():
    hold, count, rate = wrapped_rate_motion_hold(
        current_delta_rad=np.deg2rad(5.0),
        previous_delta_rad=0.0,
        dt_s=0.1,
        hold_active=False,
        release_count=0,
        enter_rate_rad_s=np.deg2rad(3.0),
        release_rate_rad_s=np.deg2rad(1.0),
        release_samples=3,
        observation_valid=True,
    )
    assert hold and count == 0 and rate > 0.0

    previous = np.deg2rad(5.0)
    for expected_count in (1, 2):
        hold, count, _ = wrapped_rate_motion_hold(
            current_delta_rad=previous,
            previous_delta_rad=previous,
            dt_s=0.1,
            hold_active=hold,
            release_count=count,
            enter_rate_rad_s=np.deg2rad(3.0),
            release_rate_rad_s=np.deg2rad(1.0),
            release_samples=3,
            observation_valid=True,
        )
        assert hold and count == expected_count
    hold, count, _ = wrapped_rate_motion_hold(
        current_delta_rad=previous,
        previous_delta_rad=previous,
        dt_s=0.1,
        hold_active=hold,
        release_count=count,
        enter_rate_rad_s=np.deg2rad(3.0),
        release_rate_rad_s=np.deg2rad(1.0),
        release_samples=3,
        observation_valid=True,
    )
    assert not hold and count == 0


def test_wrist_motion_hold_requires_semantic_arming_only_for_new_latch():
    common = dict(
        current_delta_rad=np.deg2rad(2.0),
        previous_delta_rad=0.0,
        dt_s=0.1,
        release_count=0,
        enter_rate_rad_s=np.deg2rad(3.0),
        release_rate_rad_s=np.deg2rad(1.0),
        release_samples=3,
        observation_valid=False,
    )

    hold, count, rate = wrapped_rate_motion_hold(
        hold_active=False,
        activation_enabled=False,
        **common,
    )
    assert not hold and count == 0 and rate > 0.0

    hold, count, _ = wrapped_rate_motion_hold(
        hold_active=False,
        activation_enabled=True,
        **common,
    )
    assert hold and count == 0

    hold, count, _ = wrapped_rate_motion_hold(
        hold_active=True,
        activation_enabled=False,
        **common,
    )
    assert hold and count == 0


def test_wrist_hold_release_is_not_vetoed_by_real_vertical_motion():
    # A real vertical gesture must not create a wrist hold by itself.
    assert wrist_height_release_ready(True, False, -0.080, 0.012)
    # Once wrist rotation has latched the hold, wrist-angle confirmation owns
    # release timing. A large real downward rate must not keep rebasing height
    # forever and lock the arm at its upper lift limit.
    assert wrist_height_release_ready(True, True, -0.080, 0.012)
    assert wrist_height_release_ready(True, True, -0.006, 0.012)
    assert not wrist_height_release_ready(False, True, 0.0, 0.012)


def test_high_vertical_rate_trace_still_releases_after_stable_wrist_samples():
    hold = True
    count = 0
    angle = np.deg2rad(20.0)
    for rate_m_s in (-0.080, -0.060, -0.040, -0.020):
        observation_valid = wrist_height_release_ready(
            True, hold, rate_m_s, 0.012
        )
        hold, count, _ = wrapped_rate_motion_hold(
            current_delta_rad=angle,
            previous_delta_rad=angle,
            dt_s=0.17,
            hold_active=hold,
            release_count=count,
            enter_rate_rad_s=np.deg2rad(3.0),
            release_rate_rad_s=np.deg2rad(1.0),
            release_samples=4,
            observation_valid=observation_valid,
            activation_enabled=False,
        )

    assert not hold and count == 0


def test_stationary_planar_target_requires_consecutive_quiet_samples():
    count = 0
    for expected_count in (1, 2):
        active, count = confirmed_stationary_planar_target(
            requested_target_m=[0.0208, 0.0600],
            current_target_m=[0.0200, 0.0600],
            current_velocity_m_s=[0.0010, 0.0],
            target_deadband_m=[0.0025, 0.0030],
            velocity_deadband_m_s=[0.0030, 0.0030],
            previous_count=count,
            confirm_samples=3,
        )
        assert not active and count == expected_count

    active, count = confirmed_stationary_planar_target(
        requested_target_m=[0.0208, 0.0600],
        current_target_m=[0.0200, 0.0600],
        current_velocity_m_s=[0.0010, 0.0],
        target_deadband_m=[0.0025, 0.0030],
        velocity_deadband_m_s=[0.0030, 0.0030],
        previous_count=count,
        confirm_samples=3,
    )
    assert active and count == 3


def test_stationary_planar_target_releases_immediately_for_new_motion():
    active, count = confirmed_stationary_planar_target(
        requested_target_m=[0.035, 0.040],
        current_target_m=[0.020, 0.060],
        current_velocity_m_s=[0.0, 0.0],
        target_deadband_m=[0.0025, 0.0030],
        velocity_deadband_m_s=[0.0030, 0.0030],
        previous_count=3,
        confirm_samples=3,
    )
    assert not active and count == 0


def test_wrist_hold_cancels_queued_vertical_descent_velocity():
    target, velocity = wrist_hold_vertical_planner_state(
        requested_lift_m=0.025,
        current_command_m=0.050,
        previous_velocity_m_s=-0.020,
        hold_active=True,
    )
    assert target == pytest.approx(0.050)
    assert velocity == 0.0


def test_static_task_feedback_cannot_lower_protected_frames():
    vertical_rows = np.array([[1.0, 0.0], [0.5, 1.0]])
    guarded = suppress_uncommanded_task_descent(
        joint_velocity=[-0.20, -0.10],
        vertical_task_rows=vertical_rows,
        descent_allowed=False,
    )
    deliberate = suppress_uncommanded_task_descent(
        joint_velocity=[-0.20, -0.10],
        vertical_task_rows=vertical_rows,
        descent_allowed=True,
    )

    assert np.all(vertical_rows @ guarded >= -1e-10)
    np.testing.assert_allclose(deliberate, [-0.20, -0.10])


def test_descent_guard_allows_planar_articulation_above_table_safety_band():
    rows = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    selected, mask = select_descent_guard_rows(
        rows,
        frame_ids=[30, 40, 50],
        end_frame_id=50,
        current_heights_m=[0.030, 0.025, 0.040],
        table_heights_m=[0.0, 0.0, 0.0],
        clearance_margin_m=0.010,
    )
    np.testing.assert_array_equal(mask, [False, False, True])

    # Lowering one elevated intermediate link while raising another can keep
    # the TCP level and is necessary for normal forward/backward motion.
    requested = np.array([-0.2, 0.2])
    guarded = suppress_uncommanded_task_descent(requested, selected, False)
    np.testing.assert_allclose(guarded, requested)


def test_descent_guard_adds_intermediate_link_inside_table_safety_band():
    rows = np.eye(3)
    _, mask = select_descent_guard_rows(
        rows,
        frame_ids=[30, 40, 50],
        end_frame_id=50,
        current_heights_m=[0.006, 0.025, 0.040],
        table_heights_m=[0.0, 0.0, 0.0],
        clearance_margin_m=0.010,
    )
    np.testing.assert_array_equal(mask, [True, False, True])


def test_wrist_twist_direction_requires_confirmation_and_holds_through_spike():
    state = (0, 0.0, 0, 0)

    for _ in range(3):
        state = confirm_directional_wrist_twist(
            np.deg2rad(20.0),
            *state,
            activate_rad=np.deg2rad(12.0),
            release_rad=np.deg2rad(6.0),
            confirm_samples=4,
        )
        assert state[0] == 0
    state = confirm_directional_wrist_twist(
        np.deg2rad(20.0),
        *state,
        activate_rad=np.deg2rad(12.0),
        release_rad=np.deg2rad(6.0),
        confirm_samples=4,
    )
    assert state[0] == 1
    assert np.rad2deg(state[1]) == pytest.approx(20.0)

    state = confirm_directional_wrist_twist(
        np.deg2rad(-25.0),
        *state,
        activate_rad=np.deg2rad(12.0),
        release_rad=np.deg2rad(6.0),
        confirm_samples=4,
    )
    assert state[0] == 1
    assert np.rad2deg(state[1]) == pytest.approx(20.0)
    assert state[2:] == (-1, 1)

    for _ in range(3):
        state = confirm_directional_wrist_twist(
            np.deg2rad(-25.0),
            *state,
            activate_rad=np.deg2rad(12.0),
            release_rad=np.deg2rad(6.0),
            confirm_samples=4,
        )
    assert state[0] == -1
    assert np.rad2deg(state[1]) == pytest.approx(-25.0)

    for _ in range(4):
        state = confirm_directional_wrist_twist(
            0.0,
            *state,
            activate_rad=np.deg2rad(12.0),
            release_rad=np.deg2rad(6.0),
            confirm_samples=4,
        )
    assert state == (0, 0.0, 0, 0)


def test_wrist_twist_reversal_can_require_longer_confirmation():
    state = (1, np.deg2rad(20.0), 0, 0)
    for _ in range(6):
        state = confirm_directional_wrist_twist(
            np.deg2rad(-25.0),
            *state,
            activate_rad=np.deg2rad(12.0),
            release_rad=np.deg2rad(6.0),
            confirm_samples=4,
            reverse_confirm_samples=7,
        )
        assert state[0] == 1
    state = confirm_directional_wrist_twist(
        np.deg2rad(-25.0),
        *state,
        activate_rad=np.deg2rad(12.0),
        release_rad=np.deg2rad(6.0),
        confirm_samples=4,
        reverse_confirm_samples=7,
    )
    assert state[0] == -1


def test_weighted_planar_dls_biases_motion_toward_more_mobile_joint():
    delta = weighted_damped_least_squares(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        [1.0, 1.0],
        damping=1e-6,
        joint_mobility=[1.0, 2.0, 1.0],
    )

    np.testing.assert_allclose(delta, [0.2, 0.8, 0.2], atol=1e-8)
    np.testing.assert_allclose(
        np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]) @ delta,
        [1.0, 1.0],
        atol=1e-8,
    )


def test_horizontal_radial_axis_does_not_mutate_tcp_reference():
    tcp = np.array([0.03, -0.26, 0.19])
    original = tcp.copy()

    axis, radius = horizontal_radial_axis(tcp)

    np.testing.assert_allclose(tcp, original)
    np.testing.assert_allclose(axis, original[:2] / np.linalg.norm(original[:2]))
    assert radius == pytest.approx(np.linalg.norm(original[:2]))


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


def test_table_clearance_projection_keeps_largest_safe_motion_fraction():
    follower = object.__new__(B601PalmFollower)
    follower._semantic_table_clearance_ok = (
        lambda current, target: float(np.asarray(target)[0]) <= 0.40
    )

    projected, fraction = B601PalmFollower._project_semantic_target_to_table_clearance(
        follower,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.5]),
    )

    assert 0.399 < fraction <= 0.40
    np.testing.assert_allclose(projected, fraction * np.array([1.0, 0.5]))


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
