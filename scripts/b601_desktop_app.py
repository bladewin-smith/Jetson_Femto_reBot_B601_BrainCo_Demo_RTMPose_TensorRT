#!/usr/bin/env python3
"""Fail-closed desktop launcher for the B601 vision and BrainCo hand demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arm_control.rebot_b601 import (
    StartupGuardError,
    validate_startup_joint_limits,
    validate_tabletop_startup,
)


DEFAULT_ENGINE_DIR = REPOSITORY_ROOT / "models" / "tensorrt"
DEFAULT_SDK_ROOT = REPOSITORY_ROOT / "third_party" / "reBotArm_control_py"
DEFAULT_HARDWARE_YAML = REPOSITORY_ROOT / "config" / "rebotarm_dm_brainco.yaml"
DEFAULT_URDF = (
    DEFAULT_SDK_ROOT / "urdf" / "DM" / "urdf" / "ReBot_Arm_DM.urdf"
)
SCAN_RESULT_PREFIX = "B601_SCAN_RESULT_JSON="

# Seeed's published B601-DM range, intersected with the official URDF by the
# shared validator below.  Keep these values identical to the runtime defaults.
PUBLISHED_SAFE_LOWER_RAD = np.asarray(
    [-2.6005, -3.5954, -3.5954, -1.4992, -1.4992, -1.4992],
    dtype=np.float64,
)
PUBLISHED_SAFE_UPPER_RAD = np.asarray(
    [2.6005, 0.0, 0.0, 1.4992, 1.4992, 1.4992],
    dtype=np.float64,
)

# Joint limits alone cannot prove that an unbraked arm is resting on a table.
# In the validated table-flat geometry J2/J3/J4 are close to zero.  This extra
# envelope rejects an arbitrary unsupported pose even when it is within the
# broad official motion range. J1 yaw and J5/J6 tool orientation stay free.
TABLE_FLAT_JOINTS = np.asarray([1, 2, 3], dtype=np.int64)
TABLE_FLAT_MAX_ABS_RAD = np.asarray([0.18, 0.18, 0.25], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        default="[1.0,1.0,0.0,0.0,1.0,1.0]",
        help="b601_semantic_channel_enable passed to the main launcher",
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--interval", type=float, default=0.04)
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument("--hardware-yaml", type=Path, default=DEFAULT_HARDWARE_YAML)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--scan-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_table_flat_geometry(positions_rad: Iterable[float]) -> None:
    """Reject a broad-range pose that is not plausibly table-flat."""

    positions = np.asarray(positions_rad, dtype=np.float64).reshape(-1)
    if positions.size != 6 or not np.all(np.isfinite(positions)):
        raise ValueError("table-pose feedback must contain six finite joints")
    outside = (
        np.abs(positions[TABLE_FLAT_JOINTS]) > TABLE_FLAT_MAX_ABS_RAD
    )
    if np.any(outside):
        joints = (TABLE_FLAT_JOINTS[np.flatnonzero(outside)] + 1).tolist()
        raise StartupGuardError(
            "Refusing to enable B601-DM torque: the measured pose is inside the "
            "broad joint range but is not inside the conservative table-flat "
            f"startup envelope for joints {joints}. "
            f"positions_rad={positions.round(5).tolist()}"
        )


def required_engines(engine_dir: Path) -> dict[str, Path]:
    directory = engine_dir.expanduser().resolve()
    return {
        "DET_ENGINE": directory / "rtmdet_nano.engine",
        "BODY_ENGINE": directory / "rtmpose_body.engine",
        "HAND_ENGINE": directory / "rtmpose_hand.engine",
    }


def validate_engine_files(engine_paths: dict[str, Path]) -> None:
    missing = [str(path) for path in engine_paths.values() if not path.is_file()]
    empty = [str(path) for path in engine_paths.values() if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if empty:
            details.append(f"empty={empty}")
        raise RuntimeError(
            "Verified TensorRT engines are not installed ("
            + ", ".join(details)
            + "). Run scripts/install_b601_desktop_app.sh again."
        )
    directories = {path.parent for path in engine_paths.values()}
    if len(directories) != 1:
        raise RuntimeError("Verified TensorRT engines must share one directory")
    manifest = directories.pop() / "verified_engines.sha256"
    if not manifest.is_file():
        raise RuntimeError(
            f"Engine checksum manifest is missing: {manifest}. "
            "Run scripts/install_b601_desktop_app.sh again."
        )
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0].lower()
    for path in engine_paths.values():
        expected_hash = expected.get(path.name)
        if expected_hash is None:
            raise RuntimeError(f"Checksum manifest has no entry for {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(
                f"TensorRT engine checksum mismatch: {path}. "
                "Run scripts/install_b601_desktop_app.sh again."
            )


def scan_table_pose(
    sdk_root: Path,
    hardware_yaml: Path,
    urdf_path: Path,
    samples: int,
    interval_s: float,
) -> tuple[np.ndarray, float, float]:
    """Read and validate feedback without ever enabling motor torque."""

    if samples < 3:
        raise ValueError("at least three feedback samples are required")
    sdk_root = sdk_root.expanduser().resolve()
    hardware_yaml = hardware_yaml.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    for label, path in (
        ("B601 SDK", sdk_root),
        ("hardware YAML", hardware_yaml),
        ("DM URDF", urdf_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))

    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.kinematics import load_robot_model

    model = load_robot_model(str(urdf_path))
    model_lower = np.asarray(model.lowerPositionLimit[:6], dtype=np.float64)
    model_upper = np.asarray(model.upperPositionLimit[:6], dtype=np.float64)
    robot = RebotArm(str(hardware_yaml))
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    try:
        robot.connect()
        for _ in range(samples):
            q, dq, _ = robot.get_state(request_feedback=True)
            states = [
                robot._motor_map[name].get_state()
                for name in robot.arm.joint_names
            ]
            q_array = np.asarray(q[:6], dtype=np.float64)
            dq_array = np.asarray(dq[:6], dtype=np.float64)
            if (
                len(q) >= 6
                and len(dq) >= 6
                and all(state is not None for state in states)
                and np.all(np.isfinite(q_array))
                and np.all(np.isfinite(dq_array))
            ):
                positions.append(q_array)
                velocities.append(dq_array)
            time.sleep(max(0.01, interval_s))
    finally:
        # No mode_mit(), enable_all(), or command call is made above.
        robot.disconnect()

    minimum_samples = max(3, samples // 2)
    if len(positions) < minimum_samples:
        raise StartupGuardError(
            "Not enough complete six-joint feedback samples. Torque was not "
            "enabled. Check /dev/ttyACM0, 24 V, MotorBridge, and motor/feedback "
            "IDs 0x01..0x06/0x11..0x16."
        )

    position_matrix = np.vstack(positions)
    velocity_matrix = np.vstack(velocities)
    q_table = np.median(position_matrix, axis=0)
    peak_velocity_by_joint = np.max(np.abs(velocity_matrix), axis=0)
    max_velocity = float(np.max(peak_velocity_by_joint))
    max_spread = float(np.max(np.ptp(position_matrix, axis=0)))
    validate_tabletop_startup(
        q_table,
        peak_velocity_by_joint,
        q_table,
        [0.12] * 6,
        0.20,
    )
    if max_spread > 0.03:
        raise StartupGuardError(
            "Refusing to use a moving/unstable table pose: feedback spread was "
            f"{max_spread:.4f} rad (limit 0.0300 rad)."
        )
    validate_startup_joint_limits(
        q_table,
        model_lower,
        model_upper,
        PUBLISHED_SAFE_LOWER_RAD,
        PUBLISHED_SAFE_UPPER_RAD,
    )
    validate_table_flat_geometry(q_table)
    return q_table, max_velocity, max_spread


def parse_scan_worker_output(output: str) -> dict[str, object]:
    """Extract the final structured result from a noisy hardware subprocess."""

    for line in reversed(output.splitlines()):
        if line.startswith(SCAN_RESULT_PREFIX):
            payload = json.loads(line[len(SCAN_RESULT_PREFIX) :])
            if not isinstance(payload, dict):
                break
            return payload
    raise RuntimeError("B601 feedback scan worker returned no structured result")


def scan_worker_main(args: argparse.Namespace) -> int:
    """Scan in a disposable process so native serial handles cannot survive."""

    try:
        q_table, max_velocity, max_spread = scan_table_pose(
            args.sdk_root,
            args.hardware_yaml,
            args.urdf,
            args.samples,
            args.interval,
        )
        payload: dict[str, object] = {
            "ok": True,
            "q_table": q_table.tolist(),
            "max_velocity": max_velocity,
            "max_spread": max_spread,
        }
        return_code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return_code = 2
    print(SCAN_RESULT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)
    return return_code


def scan_table_pose_isolated(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    """Run the read-only scan in a process that fully releases MotorBridge."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scan-worker",
        "--samples",
        str(args.samples),
        "--interval",
        str(args.interval),
        "--sdk-root",
        str(args.sdk_root),
        "--hardware-yaml",
        str(args.hardware_yaml),
        "--urdf",
        str(args.urdf),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        payload = parse_scan_worker_output(completed.stdout)
    except Exception as exc:
        diagnostic = completed.stdout.strip()
        raise RuntimeError(
            f"B601 feedback scan process failed (exit={completed.returncode}): "
            f"{diagnostic or exc}"
        ) from exc
    if not bool(payload.get("ok")):
        error_type = str(payload.get("error_type", "StartupGuardError"))
        error = str(payload.get("error", "unknown feedback scan failure"))
        raise StartupGuardError(f"{error_type}: {error}")
    q_table = np.asarray(payload["q_table"], dtype=np.float64).reshape(-1)
    if q_table.size != 6 or not np.all(np.isfinite(q_table)):
        raise RuntimeError("B601 feedback scan returned an invalid six-joint pose")
    # subprocess.run() has already reaped the worker. This short bounded delay
    # also lets the USB CDC driver finish its final close before the controller
    # process opens /dev/ttyACM0 again.
    time.sleep(0.35)
    return (
        q_table,
        float(payload["max_velocity"]),
        float(payload["max_spread"]),
    )


def read_action(prompt: str, allowed: set[str]) -> str:
    """Read one key on Linux terminals, with a line-input fallback."""

    print(prompt, end="", flush=True)
    if sys.stdin.isatty():
        try:
            import termios
            import tty

            descriptor = sys.stdin.fileno()
            previous = termios.tcgetattr(descriptor)
            try:
                tty.setraw(descriptor)
                while True:
                    key = sys.stdin.read(1).lower()
                    if key == "\x03":
                        print()
                        raise KeyboardInterrupt
                    if key in allowed:
                        print(key)
                        return key
            finally:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        except (ImportError, OSError):
            pass
    while True:
        key = input().strip().lower()[:1]
        if key in allowed:
            return key


def print_banner() -> None:
    print("\n" + "=" * 72)
    print(" reBot Arm B601-DM 视觉机械臂 + BrainCo 灵巧手安全启动器")
    print("=" * 72)
    print("警告：机械臂电机失能时可能坠落。扫描前必须让整臂平行、")
    print("完整支撑在桌面上，清空工作空间，并把急停放在手边。")
    print("姿态扫描只读取反馈，绝不会使能电机。\n")


def run_control(
    q_table: np.ndarray,
    channels: str,
    engine_paths: dict[str, Path],
) -> int:
    env = os.environ.copy()
    env["TABLE_Q"] = json.dumps(
        [round(float(value), 6) for value in q_table],
        separators=(",", ":"),
    )
    for name, path in engine_paths.items():
        env[name] = str(path)
    env["B601_USE_VERIFIED_LOCAL_ENGINES"] = "1"
    # Re-check close to the freshly scanned pose inside the real controller,
    # closing the scan-to-enable race without changing the manual launcher.
    env["B601_TABLETOP_JOINT_TOLERANCE_RAD"] = (
        "[0.04,0.04,0.04,0.04,0.04,0.04]"
    )
    launcher = REPOSITORY_ROOT / "scripts" / "run_b601_smooth_follow.sh"
    return subprocess.run(
        ["bash", str(launcher), channels],
        cwd=REPOSITORY_ROOT,
        env=env,
        check=False,
    ).returncode


def main() -> int:
    args = parse_args()
    if args.scan_worker:
        return scan_worker_main(args)
    engine_paths = required_engines(args.engine_dir)
    print_banner()
    try:
        validate_engine_files(engine_paths)
    except Exception as exc:
        print(f"[模型检查失败] {exc}", file=sys.stderr)
        read_action("按 q 退出：", {"q"})
        return 2

    while True:
        print("[安全扫描] 正在读取六关节反馈（电机保持失能）……")
        try:
            q_table, max_velocity, max_spread = scan_table_pose_isolated(args)
        except Exception as exc:
            print("\n[启动被安全锁拒绝] 电机没有使能。", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            print("请断电并重新把机械臂平行、完整地放在桌面上。")
            action = read_action("摆放稳定后按 r 重新扫描；按 q 退出：", {"r", "q"})
            if action == "q":
                return 2
            print()
            continue

        values = ",".join(f"{value:.6f}" for value in q_table)
        print("[安全扫描通过]")
        print(f"TABLE_Q=[{values}]")
        print(f"max_abs_velocity={max_velocity:.6f} rad/s")
        print(f"feedback_spread={max_spread:.6f} rad")
        print("即将启动视觉、机械臂和灵巧手；识别稳定后仍需按 S 开始跟随。\n")
        return_code = run_control(q_table, args.channels, engine_paths)
        if return_code in (0, 130, -2):
            return 0 if return_code == 0 else return_code
        print(f"\n[程序异常退出] exit_code={return_code}", file=sys.stderr)
        action = read_action(
            "确认机械臂仍由桌面支撑后，按 r 重新扫描并启动；按 q 退出：",
            {"r", "q"},
        )
        if action == "q":
            return return_code
        print()


if __name__ == "__main__":
    raise SystemExit(main())
