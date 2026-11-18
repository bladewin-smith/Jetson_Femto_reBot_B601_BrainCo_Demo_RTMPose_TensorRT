# Jetson Femto + reBot Arm B601-DM + BrainCo + RTMPose TensorRT

本仓库把原 `Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT` 的 RM65-B
控制链替换为 Seeed Studio reBot Arm B601-DM，同时保留 Orbbec RGB-D、
RTMDet/RTMPose TensorRT、最近操作者选择、双手关键点与 BrainCo Revo2 手指控制。

> **安全警告：B601-DM 断电或控制链中断时不能依靠软件保持姿态。首次上电前，
> 必须把机械臂平行、完整支撑在桌面上，并确保失能时不会坠落、夹伤或撞击。
> 软件的回桌轨迹无法覆盖停电、USB 断连、进程崩溃或急停。真机操作必须有人在
> 急停旁监护，并建议使用机械支撑和 UPS。**

## 迁移后的控制链

```text
Orbbec RGB-D
  -> RTMDet 最近操作者
  -> RTMPose 身体 + 双手
  -> 手掌三维位移
  -> B601 DM URDF / Pinocchio 逆解
  -> 关节限位、单步限幅、桌面运动包络
  -> 500 Hz MIT 位置/阻尼 + Pinocchio g(q) 重力前馈
  -> MotorBridge /dev/ttyACM0 / 达妙电机 0x01..0x06

手指弯曲度
  -> BrainCo Revo2
  -> 独立 USB-RS485 或 CANFD（不再经过 RM 工具口）
```

与 RM65-B 版本的关键差异、机械参数来源和迁移决策见
[`docs/MECHANICAL_AND_CONTROL_MIGRATION.md`](docs/MECHANICAL_AND_CONTROL_MIGRATION.md)。

## 关键安全门

- `robot_motion_enabled` 默认 `false`。
- 真机启用还必须显式设置 `b601_tabletop_startup_confirmed:=true`。
- 上电后先读取六轴反馈；任一轴无反馈、机械臂仍在移动或偏离已标定桌面姿态，
  都会在使能力矩前失败退出。
- 禁止启动自动回零；初始目标始终是刚读取到的真实姿态，不是零向量。
- 桌面参考点的负 Z 位移默认是 0；抬升不足 20 mm 时禁止 XY 平移。
- 每个笛卡尔目标都重新做 B601 URDF 逆解、关节限位检查和关节跳变量检查。
- `H` 键和正常退出使用低速 minimum-jerk 轨迹回到桌面支撑姿态，验证到位后
  才正常失能。真实断电仍不受此流程保护。

## 主要文件

| 路径 | 作用 |
|---|---|
| `scripts/camera_arm_b601_control_pose_hands.py` | 完整 ROS/Orbbec/RTMPose/B601/BrainCo 入口 |
| `arm_control/rebot_b601.py` | B601 启动门、Pinocchio IK、MIT + `g(q)`、桌面回位 |
| `config/rebotarm_dm_brainco.yaml` | 仅六轴 DM 电机；刻意不注册 Seeed 夹爪 0x07 |
| `scripts/read_b601_table_pose.py` | 不使能电机，只读取并输出桌面姿态参数 |
| `vision_backends/` | TensorRT 10、RTMDet/RTMPose、深度选人和手部分割 |
| `hand_control/` | BrainCo 手指标定、迟滞、防抖与拇指对掌 |
| `third_party/reBotArm_control_py` | Seeed 官方 SDK Git submodule |
| `jetson_autostart/` | 带双重真机确认的 Jetson 启动脚本和环境模板 |

## 安装

Seeed SDK 当前声明 Python `>=3.10,<3.12`，Jetson 侧建议 Python 3.10：

```bash
git clone --recurse-submodules <this-repository-url>
cd Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT
git submodule update --init --recursive

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e third_party/reBotArm_control_py
python -m pip install -r deploy/requirements-jetson-runtime.txt
```

另外需要：ROS 2 Humble、`pyorbbecsdk`、目标 JetPack 自带 TensorRT/CUDA，及
BrainCo 官方手部 SDK（启用灵巧手时）。不要用 pip 覆盖 JetPack 的 TensorRT。

TensorRT engine 必须在目标 Jetson/JetPack/TensorRT 组合上构建；流程见
[`deploy/README.md`](deploy/README.md)。

## 分阶段验收

先定义路径：

```bash
REPO="$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
DET_ENGINE="$REPO/models/tensorrt/rtmdet_nano.engine"
BODY_ENGINE="$REPO/models/tensorrt/rtmpose_body.engine"
HAND_ENGINE="$REPO/models/tensorrt/rtmpose_hand.engine"
cd "$REPO"
source /opt/ros/humble/setup.bash
export PYTHONPATH="$HOME/ws/pyorbbecsdk-2-main:$HOME/ws/brainco-hand-sdk-main/python:$REPO:$REPO/scripts:${PYTHONPATH:-}"
```

### 1. 纯视觉，所有执行器关闭

```bash
python scripts/camera_arm_b601_control_pose_hands.py --ros-args \
  -p vision_backend:=rtmpose \
  -p rtmpose_runtime:=tensorrt \
  -p rtmdet_engine_path:="$DET_ENGINE" \
  -p rtmpose_body_engine_path:="$BODY_ENGINE" \
  -p rtmpose_hand_engine_path:="$HAND_ENGINE" \
  -p robot_motion_enabled:=false \
  -p hand_control_enabled:=false \
  -p auto_start_on_ready:=false
```

### 2. 标定“平行且由桌面支撑”的启动姿态

断电摆好并支撑机械臂，清空周围空间。接通 24 V 和 MotorBridge 后只读取反馈：

```bash
python scripts/read_b601_table_pose.py
```

把输出的六个弧度值保存为 `b601_tabletop_joints_rad`。不要把悬空姿态或仍在运动
时的读数作为桌面姿态。

### 3. 灵巧手独立验收

保持 `robot_motion_enabled:=false`，将 `hand_control_enabled:=true`，并选择：

```text
hand_transport:=brainco-rs485   # 独立 USB-RS485
hand_transport:=brainco-canfd   # 独立 SocketCAN CANFD
```

`realman-hand` 与 `raw-modbus` 已被主动拒绝，因为 B601 没有 RM 控制器工具口。

### 4. B601 小范围真机验收

确认机械臂仍完整支撑在桌面、六轴读数与标定一致、急停可用，再运行：

```bash
TABLE_Q='[q1,q2,q3,q4,q5,q6]'  # 替换为第 2 步实测弧度值

python scripts/camera_arm_b601_control_pose_hands.py --ros-args \
  -p vision_backend:=rtmpose \
  -p rtmpose_runtime:=tensorrt \
  -p rtmdet_engine_path:="$DET_ENGINE" \
  -p rtmpose_body_engine_path:="$BODY_ENGINE" \
  -p rtmpose_hand_engine_path:="$HAND_ENGINE" \
  -p robot_motion_enabled:=true \
  -p b601_tabletop_startup_confirmed:=true \
  -p b601_tabletop_joints_rad:="$TABLE_Q" \
  -p b601_tabletop_joint_tolerance_rad:='[0.12,0.12,0.12,0.12,0.12,0.12]' \
  -p auto_home_on_start:=false \
  -p max_arm_axis_offset_m:='[0.025,0.025,0.060]' \
  -p b601_workspace_negative_axis_offset_m:='[0.025,0.025,0.0]' \
  -p b601_workspace_positive_axis_offset_m:='[0.025,0.025,0.060]' \
  -p b601_min_lift_before_xy_m:=0.020 \
  -p hand_control_enabled:=false \
  -p auto_start_on_ready:=false
```

程序启动后不会立刻跟随。人员识别稳定后按 `S` 才开始；按 `H` 返回桌面姿态并
进入必须再次按 `S` 的暂停状态；`Q`/`Esc` 正常退出。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m py_compile arm_control/rebot_b601.py scripts/camera_arm_b601_control_pose_hands.py
python -m pytest tests -q
```

单元测试覆盖启动姿态拒绝、运动中拒绝、桌面 Z/XY 包络和 minimum-jerk 端点。
真机通信、重力模型符号、桌面姿态、相机外参和负载变化必须在目标 B601 上按上述
阶段重新验收，不能由无硬件测试替代。
