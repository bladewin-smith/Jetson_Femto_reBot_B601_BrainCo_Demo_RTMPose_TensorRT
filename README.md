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
| 手部整体左右位移 | J1 底座旋转 | 解剖腕点的相机 X 位移直接映射到 J1 |
| 手部整体前后位移 | J2 + J3 + J4 | 解剖腕点深度沿当前底座到 TCP 的水平径向做加权阻尼雅可比解算 |
| 手部整体上下位移 | J2 + J3 + J4 | 解剖腕点高度与前后位移组成二维任务，同时保持中间连杆净空 |
| 肘关节屈伸、腕关节俯仰 | 不直接控制关节 | 仍可用于视觉诊断，但不再分别映射到 J3/J4 |
| 前臂旋前/旋后 | J6 | 三维掌面法向量绕肘—腕轴的确认角映射到 J6 |

J5 在该模式下保持本次跟踪开始时的参考角。平移信号不再使用腕点与多个 MCP 点的
掌心质心，而是优先使用人体姿态模型的解剖腕点，并以手部腕点作深度一致性检查和
回退。这样纯粹旋腕时，指根绕腕点转动不会被误识别成手掌上下/前后平移。二维解算器
把腕点深度和高度作为一个
连续目标，通过 `b601_semantic_planar_joint_mobility` 调整 J2/J3/J4 的相对参与度，
不再依赖“肘是否前移”或“腕是否前倾”的判断边界。垂直通道只允许从经过验证的桌面姿态
向上运动；手掌下移表示回落到桌面参考高度，不能命令末端进入桌面以下。控制器还会
比较 URDF 中所有 frame 与启动桌面姿态的高度，并拒绝降低到容差以下的目标。任一
相机轴上的掌心位移超过 `b601_semantic_max_observation_displacement_m`（默认 0.35m）
时视为跟踪离群值，控制器会保持实测关节姿态并重置视觉滤波器。

六个 `b601_semantic_channel_enable` 值依次表示：
`[J1左右, J2/J3/J4前后, 保留, 保留, J6滚转, J2/J3/J4上下]`。第 3、4 项仅为兼容
旧命令保留，设成 0 或 1 都不会再把肘/腕角送给电机。为了避免贴桌拖动，前后与 J1
会根据 TCP 实际抬升和连杆实际净空从 0 到 1 连续淡入；J6 在净空达到 95% 后才解锁，
并在解锁瞬间重新采集腕部零位，避免把抬手动作误判成旋腕。上下通道始终可用于离桌。
高度任务分别给 link3、link4、TCP 分配 50%、82%、100%
的目标抬升量，避免只抬末端而让中间连杆继续贴桌。每次控制手左右切换也会重新采集
当前人体掌心和关节参考，避免瞬时跳变。

J4 另外使用两级抑振：任务空间的前后/上下目标先按最大速度连续爬坡，随后 J4
关节目标再经过小死区、最大速度和最大加速度限制。明显运动配置把 J4 的 MIT 参数从
高刚度改为 `kp=30、kd=4`，并降低 J4 在冗余解算中的 mobility，避免视觉深度跳变
或雅可比分配变化直接变成 J4 的快速正反转。

### 安全的明显运动分通道验证

必须先完成上面的桌面姿态标定。下面的第一阶段已经提高到肉眼可见但仍受约束的
验证范围：目标 TCP 抬升约 `25 mm`，J2/J3/J4 最大偏移约 `4.6°/5.7°/4.0°`，
同时把六轴 MIT 总指令力矩分别限制为 `8/10/16/6/4/4 Nm`。灵巧手保持关闭。
TensorRT 日志确认只有旧 `rtmpose_hand.engine` 在当前板卡上触发跨设备 plan 警告；
既有 detector 与 body engine 能稳定输出约 `0.83--0.90` 的人物置信度。默认只重建
hand engine，避免未经视觉验证的新 detector 取代已知正常的检测器：

```bash
cd "$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
export SOURCE_ONNX_DIR="$HOME/ws/Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT/models/onnx"
bash scripts/build_b601_local_tensorrt.sh
```

输出位于 `models/tensorrt_local`，并记录 `build_info.txt`。后续平滑跟随脚本自动选择
“已验证 detector + 已验证 body + 本机 hand”，并打印
`Engine profile: verified-hybrid`。只有专门诊断三份本机 engine 时才设置
`B601_USE_ALL_LOCAL_ENGINES=1`；最新实测中的本机 detector 无法识别人，不能用于真机
跟随。若需要强制重建全部模型，可显式设置 `B601_ENGINE_SET=all`。

为避免旧 shell 函数继续覆盖成 12 mm 小幅参数，仓库同时提供可直接运行的明显运动
验证入口。`TABLE_Q` 必须使用本机最新读取的桌面姿态：

```bash
cd "$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
export TABLE_Q='[q1,q2,q3,q4,q5,q6]'
bash scripts/run_b601_visible_lift.sh '[0.0,0.0,0.0,0.0,0.0,1.0]'
```

若三份 TensorRT engine 不在同一目录，先分别导出已经在该 Jetson 上验证过的路径；
启动脚本会优先使用它们并打印最终选择：

```bash
export DET_ENGINE='/absolute/path/to/rtmdet_nano.engine'
export BODY_ENGINE='/absolute/path/to/rtmpose_body.engine'
export HAND_ENGINE='/absolute/path/to/rtmpose_hand.engine'
```

当前完整跟随唯一推荐使用 `run_b601_smooth_follow.sh`；不要再复用 shell 历史里的
`run_stage_v2` 或 `run_b601_semantic_test`，否则旧命令会重新覆盖成小范围参数。脚本会
显式打印最终 engine、通道和安全工作区：

```bash
cd "$HOME/ws/Jetson_Femto_reBot_B601_BrainCo_Demo_RTMPose_TensorRT"
source .venv/bin/activate
export TABLE_Q='[q1,q2,q3,q4,q5,q6]'

# 先验证 J2/J3/J4 + J6，再做完整跟随
bash scripts/run_b601_smooth_follow.sh '[0.0,0.0,0.0,0.0,1.0,1.0]'
bash scripts/run_b601_smooth_follow.sh '[1.0,1.0,0.0,0.0,1.0,1.0]'
```

Seeed Wiki 公布的 B601-DM 关节范围为 J1 `±149°`、J2/J3 `0°～-206°`、
J4/J5/J6 `±85.9°`；官方当前 DM URDF 则是 J1 `±2.8 rad`、J2/J3
`-3.14～0 rad`、J4 `-1.87～1.57 rad`、J5 `±1.57 rad`、J6 `±3.14 rad`。
运行时取两份官方资料的逐轴交集，再保留 `0.10 rad` 软件余量。J2/J3 的桌面支撑
姿态靠近 `0 rad`，因此该侧只能回到本次启动验证的 `TABLE_Q`，不能越过它。扩大后的
语义范围为 J1/J2/J3/J4/J6 相对参考最大 `1.60/1.40/1.50/1.00/1.20 rad`，
但最终仍受上述绝对包络、桌面高度投影、跟踪误差、力矩和失速保护共同限制；J5 继续
保持，不开放视觉控制。

下面的完整展开函数保留作旧版小范围诊断参考，不应用于当前完整跟随。

语义目标更新率取决于 Jetson 上的实际视觉推理周期；最新日志中位周期约
`0.189 s`（约 `5.3 Hz`）。J2/J3/J4
使用雅可比 resolved-rate 解算，将掌心任务速度和当前任务误差直接转换为协调的关节速度，不再每帧产生一个远端 IK 姿态后逐步
追赶。前后与上下通道分别执行反积分饱和；其中一个方向发生反馈积压时，不再冻结另
一个仍可安全运动的方向。桌面净空门控只限制继续向外运动，不会在净空下降时凭空制
造反向目标，且 follower mobility 连续、带滞回并受变化率限制。500 Hz MIT 层以该
关节速度为主要前馈；J2/J3/J4 的协调速度现在真正经过组内低通插值，不再被解析速度
直接覆盖。高频位置修正会根据目标领先量在流速度上限的 `18%--32%` 之间平滑变化：
靠近目标时保持柔和，J3/J4 接近跟踪窗边界时提高追赶速度，避免形成“反馈移动一点、
外层窗口再前移一点”的棘轮式上升。垂直任务反积分从实际误差 `65%` 处开始，并允许
`50 mm` 的受控任务积压；关节跟踪窗、速度、加速度、力矩和桌面保护仍是硬约束。
500 Hz 层继续执行速度、加速度和 jerk 限制。J2/J3/J4 使用独立的流速度倍率，使高频
轨迹跨越下一视觉目标周期连续运动；操作者
目标连续三帧保持静止后，外层停止冗余任务误差重解算，仅保留 MIT 内环对已接受关节
目标的保持。目标反向时会清除上一方向残留的内部速度，避免先停住、再突然追赶或短暂
反向。

掌心滤波会按运动速度自适应带宽并限制单帧跳变。J6 在圆周滤波之前增加原始角速度和
单帧角跳变拒绝；掌心快速平移、掌面深度失效或桌面净空不足时，会保持最后一个已接
受的 J6 目标并重新采集腕部零位，不会把异常帧继续积分成旋腕指令。J1/J6 的完整目标
只经过一次速度/加速度整形，再施加反馈跟踪窗口，使对应限速参数真正生效。最终目标
仍会经过关节范围、桌面投影、力矩和失速保护。

上下通道另有两层解耦。经过提速滤波的持续腕角变化只有在腕部相对零位达到 `6°`
或 J6 方向已经确认后才会触发高度冻结，避免日志中 `1--2°` 关键点噪声把高度长期
锁住；控制器持续重置
高度观测偏置，并要求腕角与解剖腕点高度漂移都连续稳定五帧，才从新腕姿继续计算真正
的上下平移，因此不会在旋腕结束后补做一次延迟下移。若平面命令没有明确的负向高度速
度，TCP 的闭环修正会被投影到不下降的速度空间；中间连杆只在距离桌面参考高度
`10 mm` 的安全带内启用同样的速度保护，安全带外允许完成前后运动所必需的小幅降高。
所有连杆仍经过完整路径的桌面高度硬投影。主动向下移动手部时速度保护自动放行，
按 `H` 回桌路径也不受影响。

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
    -p b601_semantic_joint_gains:='[0.65,-0.60,0.18,0.15,0.45]' \
    -p b601_semantic_angle_deadband_deg:='[5.0,5.0,3.0]' \
    -p palm_filter_alpha:=0.30 \
    -p palm_filter_fast_alpha:=0.72 \
    -p palm_filter_speed_scale_m_s:=0.30 \
    -p palm_filter_max_innovation_m:='[0.030,0.030,0.030]' \
    -p b601_wrist_twist_filter_alpha:=0.18 \
    -p b601_wrist_twist_max_innovation_deg:=8.0 \
    -p b601_wrist_twist_max_raw_rate_deg_s:=180.0 \
    -p b601_wrist_twist_max_raw_jump_deg:=25.0 \
    -p b601_wrist_twist_translation_gate_m_s:=0.050 \
    -p b601_wrist_twist_stationary_confirm_samples:=3 \
    -p b601_vertical_wrist_decoupling_enabled:=true \
    -p b601_vertical_wrist_hold_rate_deg_s:=3.0 \
    -p b601_vertical_wrist_hold_min_delta_deg:=6.0 \
    -p b601_vertical_wrist_release_rate_deg_s:=1.0 \
    -p b601_vertical_wrist_release_samples:=5 \
    -p b601_vertical_wrist_release_lift_rate_m_s:=0.012 \
    -p b601_wrist_twist_activate_deg:=12.0 \
    -p b601_wrist_twist_release_deg:=6.0 \
    -p b601_wrist_twist_confirm_samples:=3 \
    -p b601_wrist_twist_reverse_confirm_samples:=6 \
    -p b601_wrist_twist_min_joint_offset_rad:=0.080 \
    -p b601_wrist_twist_soft_start_s:=0.30 \
    -p b601_semantic_max_joint_offset_rad:='[0.035,0.080,0.100,0.070,0.008,0.250]' \
    -p b601_semantic_max_step_rad:='[0.0025,0.0025,0.0030,0.0020,0.0010,0.0060]' \
    -p b601_joint_kp:='[120.0,120.0,180.0,30.0,35.0,30.0]' \
    -p b601_joint_kd:='[8.0,8.0,8.0,4.0,2.0,2.0]' \
    -p b601_joint_effort_limit_nm:='[8.0,10.0,16.0,6.0,4.0,4.0]' \
    -p b601_stream_max_velocity_rad_s:='[0.25,0.30,0.35,0.15,0.10,0.25]' \
    -p b601_stream_max_acceleration_rad_s2:='[2.0,2.5,3.0,1.2,0.8,2.0]' \
    -p b601_stream_max_jerk_rad_s3:='[80.0,100.0,120.0,50.0,30.0,80.0]' \
    -p b601_stream_outer_velocity_scale:=1.40 \
    -p b601_stream_goal_velocity_filter_alpha:=0.45 \
    -p b601_stream_goal_velocity_stale_cycles:=2.0 \
    -p b601_stream_goal_period_filter_alpha:=0.20 \
    -p b601_stream_goal_velocity_max_hold_s:=0.20 \
    -p b601_stream_position_correction_gain_s:=5.0 \
    -p b601_stream_position_correction_max_fraction:=0.38 \
    -p b601_semantic_max_tracking_error_rad:='[0.012,0.020,0.055,0.018,0.008,0.040]' \
    -p b601_semantic_stall_error_rad:='[0.010,0.015,0.030,0.015,0.010,0.012]' \
    -p b601_semantic_stall_min_progress_rad:=0.0015 \
    -p b601_semantic_stall_timeout_s:=4.0 \
    -p b601_semantic_vertical_primary_joint:=3 \
    -p b601_semantic_vertical_follower_progress_lead:=0.15 \
    -p b601_semantic_vertical_max_lift_m:=0.025 \
    -p b601_semantic_min_lift_before_articulation_m:=0.018 \
    -p b601_semantic_min_chain_clearance_m:=0.001 \
    -p b601_semantic_vertical_height_profile:='[0.50,0.82,1.0]' \
    -p b601_semantic_planar_input_scale:='[0.45,1.0]' \
    -p b601_semantic_planar_max_forward_m:=0.035 \
    -p b601_semantic_planar_joint_mobility:='[0.85,1.00,0.50]' \
    -p b601_semantic_planar_task_weights:='[1.00,0.65,0.85,1.00]' \
    -p b601_semantic_planar_command_max_velocity_m_s:='[0.060,0.035]' \
    -p b601_semantic_planar_command_max_acceleration_m_s2:='[0.32,0.16]' \
    -p b601_semantic_planar_command_deadband_m:='[0.0015,0.0015]' \
    -p b601_semantic_planar_tracking_error_m:='[0.025,0.025]' \
    -p b601_semantic_planar_backpressure_start_fraction:=0.55 \
    -p b601_semantic_planar_backpressure_rates_s:='[2.0,4.0]' \
    -p b601_semantic_clearance_gate_rates_s:='[1.5,3.0]' \
    -p b601_semantic_clearance_gate_deadband:=0.03 \
    -p b601_semantic_planar_follower_min_mobility_scale:=0.30 \
    -p b601_semantic_planar_task_error_gain_s:='[2.0,2.0]' \
    -p b601_semantic_planar_task_correction_max_velocity_m_s:='[0.025,0.020]' \
    -p b601_semantic_vertical_descent_guard_enabled:=true \
    -p b601_semantic_vertical_descent_intent_deadband_m_s:=0.0015 \
    -p b601_semantic_vertical_descent_guard_clearance_margin_m:=0.010 \
    -p b601_semantic_planar_joint_max_acceleration_rad_s2:='[0.45,0.60,0.20]' \
    -p b601_semantic_motion_reverse_deadband_m:='[0.006,0.0035,0.005]' \
    -p b601_semantic_motion_reverse_confirm_samples:=3 \
    -p b601_semantic_j1_target_max_velocity_rad_s:=0.08 \
    -p b601_semantic_j1_target_max_acceleration_rad_s2:=0.35 \
    -p b601_semantic_j1_target_deadband_rad:=0.0015 \
    -p b601_semantic_j4_target_max_velocity_rad_s:=0.025 \
    -p b601_semantic_j4_target_max_acceleration_rad_s2:=0.10 \
    -p b601_semantic_j4_target_deadband_rad:=0.0015 \
    -p b601_semantic_j6_target_max_velocity_rad_s:=0.10 \
    -p b601_semantic_j6_target_max_acceleration_rad_s2:=0.40 \
    -p b601_semantic_j6_target_deadband_rad:=0.0015 \
    -p b601_semantic_linear_deadband_m:='[0.012,0.008,0.008]' \
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
`actual_error` 中的 `height` 应逐步降到约 `0.005m` 以下。机械臂抬升过程中
`forward_gate` 会从 0 平滑增加到 1；掌心前后和 J1 逐步达到完整幅度。J6 则等待
`J6_clearance=armed` 和 `J6_observation=ready`，此时会自动在抬升后的姿态重新置零；
看到这两个字段后保持掌心位置，再开始慢速旋腕。若出现 `outlier_hold`、
`translation_hold` 或 `invalid_hold`，J6 应保持当前位置，待 `J6_stable=3/3` 后重新置零。
旋腕前 `vertical_wrist_armed=False` 时，轻微角速度或无效帧不应启动高度冻结；腕角达到
`6°` 或 J6 方向确认后应变为 `vertical_wrist_armed=True`。旋腕期间
`vertical_wrist_hold=True` 时，`raw_lift` 可以随视觉几何变化，但 `lift` 应保持；
`lift_bias` 会吸收这段伪位移。腕角连续稳定 5 帧后解除保持；解除帧会把
`vertical_wrist_release` 计数清零。高度变化率只用于诊断，不能继续延长腕部保持，否则真实下移动作会被不断重写
为偏置并锁死在最大高度。手部静止时 `vertical_descent_allowed=False`，各受保护 frame 不应再
因为误差自适应而主动向下；只有明确向下移动手部时该字段才应短暂变为 `True`。
`planar_idle=True:3/3` 表示 J2/J3/J4 已进入静止目标保持；新的前后或上下动作应在下一
帧立即清零并恢复联合解算。
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

# 2. 抬起后，验证掌心整体前后 -> J2/J3/J4 联合解算
CHANNELS='[0.0,1.0,0.0,0.0,0.0,1.0]'; run_b601_semantic_test

# 3. 抬起后，屈肘并张开手掌，验证前臂旋前/旋后 -> J6
CHANNELS='[0.0,0.0,0.0,0.0,1.0,1.0]'; run_b601_semantic_test

# 4. 前三组全部通过后才做组合验证；保留通道固定为 0
CHANNELS='[1.0,1.0,0.0,0.0,1.0,1.0]'; run_b601_semantic_test
```

J6 不再使用食指 MCP 5 到小指 MCP 17 的二维连线角度。新算法使用 RTMPose 的
21 点手部拓扑和 Orbbec 深度，以腕点和 MCP 5/17（有可靠深度时再加入 MCP 9）
构造掌面法向量，并以人体
肩—肘—腕平面的法向量作为零方向，计算它绕肘—腕轴的有符号角度。这与
[MediaPipe Hands](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)
公开的 21 点三维手部表示一致，但继续复用现有
[MMPose RTMPose 手部模型](https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_hand_demo.md)
和 TensorRT 引擎，不增加第二套推理运行时。原始角速度/角跳变拒绝、圆周低通滤波、
五帧同方向确认、九帧反向确认以及 J6 独立速度/加速度限制会抑制深度尖峰。确认后的
J6 至少产生 `0.080rad`（约 4.6°）的可见目标，最大限制为 `0.250rad`（约 14.3°）；
孤立反向帧会保持原方向，不会立即反转。任一必需点没有可靠深度、掌心快速平移，或
手臂几乎完全伸直导致肩—肘—腕平面退化时，控制器保持最后一个已接受的 J6 目标并
等待稳定观测重新置零，而不是猜测。测试时肘部保持约 60–120° 弯曲，张开手掌，肩、
肘、腕和掌心均留在画面与深度量程内。`J6_raw_innovation` 是原始单帧角度变化，
`J6_translation_speed` 是掌心平移门控量，`J6_delta` 是滤波后的腕部角度，
`J6_confirmed` 是消抖确认值，`J6_offset` 是关节目标偏移，`J6_measured_offset` 是电机
反馈的真实偏移。

如果 J1 或 J6 方向与现场期望相反，只翻转 `b601_semantic_joint_gains` 中第 1 或第 5 项；
前后方向由末端径向几何定义，不应通过翻转某个 J2/J3/J4 增益来修改。若出现持续振荡、
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
