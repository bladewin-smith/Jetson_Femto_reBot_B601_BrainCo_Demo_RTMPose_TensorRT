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

## B601 人体语义关节控制

`b601_control_mode:=semantic_joint` 使用人体动作的语义，而不是把掌心位移直接交给
六轴笛卡尔 IK。当前映射为：

| 人体信号 | B601 动作 | 实现 |
|---|---|---|
| 掌心左右位移 | J1 底座旋转 | 相机 X 位移直接映射到 J1 |
| 掌心整体前后位移 | J2 | 相机深度 Z 位移映射到 J2 |
| 肘关节屈伸 | J3 | 肩-肘-腕夹角相对启动参考值映射到 J3 |
| 腕关节小幅前后俯仰 | J4 | 前臂和掌轴在相机垂直/深度平面的夹角映射到 J4 |
| 掌面旋转 | J6 | 食指/小指掌指关节连线在图像平面的滚转角映射到 J6 |
| 掌心上下位移 | J2 + J3 + J4 | 阻尼雅可比最小二乘，同时抑制末端姿态变化 |

J5 在该模式下保持本次跟踪开始时的参考角。垂直通道只允许从经过验证的桌面姿态
向上运动；手掌下移表示回落到桌面参考高度，不能命令末端进入桌面以下。控制器还会
比较 URDF 中所有 frame 与启动桌面姿态的高度，并拒绝降低到容差以下的目标。任一
相机轴上的掌心位移超过 `b601_semantic_max_observation_displacement_m`（默认 0.35m）
时视为跟踪离群值，控制器会保持实测关节姿态并重置视觉滤波器。

六个 `b601_semantic_channel_enable` 值依次表示：
`[J1左右, J2深度, J3肘, J4腕, J6滚转, J2/J3/J4上下]`。为了避免贴桌拖动，
TCP 实际抬升达到 `b601_semantic_min_lift_before_articulation_m`，并且肘部、腕部和
TCP 三个模型 frame 的最小净空达到 `b601_semantic_min_chain_clearance_m` 之前，前五个
通道保持锁定。上下控制分别给这三个 frame 分配 50%、82%、100% 的目标抬升量，
避免只抬末端而让中间连杆继续贴桌。每次控制手左右切换也会重新采集人体角度和当前
关节参考，避免瞬时跳变。

### 安全的明显运动分通道验证

必须先完成上面的桌面姿态标定。下面的第一阶段已经提高到肉眼可见但仍受约束的
验证范围：目标 TCP 抬升约 `25 mm`，J2/J3/J4 最大偏移约 `4.6°/5.7°/4.0°`，
同时把六轴 MIT 总指令力矩分别限制为 `8/10/16/6/4/4 Nm`。灵巧手保持关闭。
引擎路径使用已经在同一台
Jetson 上验收过的
RealMan 旧仓库引擎文件；不要跨 JetPack/TensorRT 版本复制 engine。

为避免旧 shell 函数继续覆盖成 12 mm 小幅参数，仓库同时提供可直接运行的明显运动
验证入口。`TABLE_Q` 必须使用本机最新读取的桌面姿态：

```bash
cd "$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
export TABLE_Q='[q1,q2,q3,q4,q5,q6]'
bash scripts/run_b601_visible_lift.sh '[0.0,0.0,0.0,0.0,0.0,1.0]'
```

该入口固定读取 `config/b601_visible_motion.yaml`，因此不要再用旧的 `run_stage_v2`
验证这一档参数。下面保留完整展开形式，便于逐项检查或继续调参。

```bash
REPO="$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
ENGINE_DIR="$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models/tensorrt"
DET_ENGINE="$ENGINE_DIR/rtmdet_nano.engine"
BODY_ENGINE="$ENGINE_DIR/rtmpose_body.engine"
HAND_ENGINE="$ENGINE_DIR/rtmpose_hand.engine"
TABLE_Q='[q1,q2,q3,q4,q5,q6]'  # 必须替换为本机刚标定的六轴弧度值

cd "$REPO"
source .venv/bin/activate
source /opt/ros/humble/setup.bash
export PYTHONPATH="$HOME/ws/pyorbbecsdk-2-main:$HOME/ws/brainco-hand-sdk-main/python:$REPO:$REPO/scripts:$REPO/third_party/reBotArm_control_py:${PYTHONPATH:-}"

for file in "$DET_ENGINE" "$BODY_ENGINE" "$HAND_ENGINE"; do
  if test -s "$file"; then
    ls -lh "$file"
  else
    echo "缺失或为空（不要继续真机启动）: $file"
  fi
done

run_b601_semantic_test() {
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
    -p b601_control_mode:=semantic_joint \
    -p b601_semantic_channel_enable:="$CHANNELS" \
    -p b601_semantic_joint_gains:='[0.65,-0.60,0.18,0.15,0.25]' \
    -p b601_semantic_max_joint_offset_rad:='[0.035,0.080,0.100,0.070,0.008,0.035]' \
    -p b601_semantic_max_step_rad:='[0.0025,0.0025,0.0030,0.0020,0.0010,0.0020]' \
    -p b601_joint_kp:='[120.0,120.0,180.0,45.0,35.0,30.0]' \
    -p b601_joint_kd:='[8.0,8.0,8.0,2.0,2.0,2.0]' \
    -p b601_joint_effort_limit_nm:='[8.0,10.0,16.0,6.0,4.0,4.0]' \
    -p b601_semantic_max_tracking_error_rad:='[0.012,0.020,0.055,0.025,0.008,0.015]' \
    -p b601_semantic_stall_error_rad:='[0.010,0.015,0.030,0.015,0.010,0.012]' \
    -p b601_semantic_stall_min_progress_rad:=0.0015 \
    -p b601_semantic_stall_timeout_s:=4.0 \
    -p b601_semantic_vertical_primary_joint:=3 \
    -p b601_semantic_vertical_follower_progress_lead:=0.15 \
    -p b601_semantic_vertical_max_lift_m:=0.025 \
    -p b601_semantic_min_lift_before_articulation_m:=0.018 \
    -p b601_semantic_min_chain_clearance_m:=0.001 \
    -p b601_semantic_vertical_height_profile:='[0.50,0.82,1.0]' \
    -p b601_semantic_vertical_solver_step_rad:=0.015 \
    -p b601_semantic_vertical_solver_iterations:=12 \
    -p b601_semantic_vertical_orientation_weight_m_per_rad:=0.02 \
    -p b601_semantic_table_clearance_tolerance_m:=0.001 \
    -p control_hand_side:=right \
    -p control_hand_fallback_enabled:=false \
    -p hand_control_enabled:=false \
    -p auto_home_on_start:=false \
    -p auto_start_on_ready:=false
}
```

每次只执行下面一组。启动后保持人体静止，按 `S` 采集参考；先把掌心缓慢抬高。
日志中的 `tcp_lift` 应比按 `S` 时增加至少约 `0.015m`，最终目标约 `0.025m`；
`chain_clearance` 应达到约 `0.001m`，并出现 `articulation=ready`；
`actual_vertical_error` 应逐步降到约 `0.005m` 以下。
`q_target` 会从上一次目标连续爬坡，并由 `target_lead` 显示它相对实测关节角积累的
有限位置误差。`q_sent` 是经过总力矩限制后真正发送的目标；`tau_g` 是重力前馈，
`tau_measured` 是电机反馈力矩，`effort_cmd` 是包含位置、阻尼与重力项的指令估算，
任何 `effort_limited=true` 表示该轴已被软件力矩上限截断。正常使能时六个
`motor_status` 均应为 `0x1`。
抬升阶段以负载最大的 J3 实测进度为主时钟，`J3_progress` 表示其已完成比例，
`follower_limit` 是 J2/J4 当时允许达到的最大比例，避免两个快关节先运动反而压低末端。
如果某关节在有足够目标误差时持续静止，看门狗会保持实测姿态并明确打印关节名；
此时不要继续加大增益，应检查该电机、接头和 24 V 电源。然后再做对应的单一小动作。
按 `H` 验证慢速回桌，确认完成后按 `Q` 退出，再进行下一组。

```bash
# 0. 先验证纯上下协同；除 J2/J3/J4 外不应有动作
CHANNELS='[0.0,0.0,0.0,0.0,0.0,1.0]'; run_b601_semantic_test

# 1. 抬起后，验证掌心左右 -> J1
CHANNELS='[1.0,0.0,0.0,0.0,0.0,1.0]'; run_b601_semantic_test

# 2. 抬起后，验证掌心整体前后 -> J2
CHANNELS='[0.0,1.0,0.0,0.0,0.0,1.0]'; run_b601_semantic_test

# 3. 抬起后，验证肘屈伸 -> J3
CHANNELS='[0.0,0.0,1.0,0.0,0.0,1.0]'; run_b601_semantic_test

# 4. 抬起后，验证腕部小幅俯仰 -> J4
CHANNELS='[0.0,0.0,0.0,1.0,0.0,1.0]'; run_b601_semantic_test

# 5. 抬起后，验证掌面旋转 -> J6
CHANNELS='[0.0,0.0,0.0,0.0,1.0,1.0]'; run_b601_semantic_test

# 6. 前六组全部通过后才做组合验证
CHANNELS='[1.0,1.0,1.0,1.0,1.0,1.0]'; run_b601_semantic_test
```

如果单个通道方向与现场期望相反，只翻转
`b601_semantic_joint_gains` 中对应项的正负号，不要同时改多个通道。若出现持续振荡、
frame 低于桌面的拒绝日志、反馈异常或实际连杆接近桌面，立即停止手势并按 `H`；若
正常按键失效，则托住机械臂并使用硬件急停。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m py_compile arm_control/rebot_b601.py scripts/camera_arm_b601_control_pose_hands.py
python -m pytest tests -q
```

单元测试覆盖启动姿态拒绝、运动中拒绝、桌面 Z/XY 包络和 minimum-jerk 端点。
真机通信、重力模型符号、桌面姿态、相机外参和负载变化必须在目标 B601 上按上述
阶段重新验收，不能由无硬件测试替代。
