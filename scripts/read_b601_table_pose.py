#!/usr/bin/env python3
"""Read a B601-DM table-supported pose without enabling any motor."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read all six B601-DM joints while torque remains disabled and print "
            "the b601_tabletop_joints_rad ROS parameter."
        )
    )
    parser.add_argument(
        "--sdk-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "reBotArm_control_py",
    )
    parser.add_argument(
        "--hardware-yaml",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "rebotarm_dm_brainco.yaml",
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--interval", type=float, default=0.04)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    sdk_root = args.sdk_root.expanduser().resolve()
    hardware_yaml = args.hardware_yaml.expanduser().resolve()
    sys.path.insert(0, str(sdk_root))

    from reBotArm_control_py.actuator import RebotArm

    print("WARNING: keep the unpowered arm fully supported and parallel to the table.")
    print("This utility reads feedback only; it never enables the motors.")
    robot = RebotArm(str(hardware_yaml))
    positions = []
    velocities = []
    try:
        robot.connect()
        for _ in range(args.samples):
            q, dq, _ = robot.get_state(request_feedback=True)
            states = [robot._motor_map[name].get_state() for name in robot.arm.joint_names]
            if len(q) >= 6 and all(state is not None for state in states):
                positions.append(np.asarray(q[:6], dtype=np.float64))
                velocities.append(np.asarray(dq[:6], dtype=np.float64))
            time.sleep(max(0.01, args.interval))
    finally:
        robot.disconnect()

    if len(positions) < max(3, args.samples // 2):
        raise RuntimeError(
            "Not enough valid six-joint feedback samples. Check /dev/ttyACM0, "
            "24 V power, MotorBridge, and motor/feedback IDs 0x01..0x06/0x11..0x16."
        )
    q_table = np.median(np.vstack(positions), axis=0)
    max_velocity = float(np.max(np.abs(np.vstack(velocities))))
    values = ",".join(f"{value:.6f}" for value in q_table)
    print(f"max_abs_velocity_rad_s={max_velocity:.6f}")
    print(f"B601_TABLETOP_JOINTS_RAD=[{values}]")
    print(f"-p b601_tabletop_joints_rad:='[{values}]'")
    if max_velocity > 0.20:
        raise RuntimeError("Arm feedback was moving; do not use this pose as the startup reference")


if __name__ == "__main__":
    main()
