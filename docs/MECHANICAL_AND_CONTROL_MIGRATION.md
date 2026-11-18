# B601-DM / RM65-B 机械与控制迁移说明

## 1. 调研依据

- Seeed 官方 `reBot-DevArm` 给出 B601-DM：6 轴 + 夹爪、1.5 kg 负载、
  767 mm 最大臂展、约 4.5 kg、24 V、重复定位精度小于 ±0.2 mm：
  <https://github.com/Seeed-Projects/reBot-DevArm>
- Seeed B601-DM Pinocchio 教程说明 `tau = g(q)` 的 MIT 重力补偿、当前位置
  目标与 `kp/kd`，并在正常退出时先保持、minimum-jerk 归零、再断开：
  <https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/>
- Seeed 官方 Python 控制 SDK：
  <https://github.com/Seeed-Projects/reBotArm_control_py>
- Realman 官方 RM65 参数给出 5 kg 负载、610/627 mm 臂展、约 7.2/7.3 kg、
  ±0.05 mm；J1-J3 为机械制动、J4-J6 为弹簧制动：
  <https://www.realman-robotics.com/en/products/rm65.html>

此外，本项目按目标硬件的明确约束处理：B601-DM 失电后电机直接失能，不能把
软件保持、正常退出轨迹或重力补偿当成断电制动器。

## 2. 结构差异及其软件后果

| 项目 | Realman RM65-B | Seeed reBot Arm B601-DM | 本仓库处理 |
|---|---|---|---|
| 驱动架构 | 独立机器人控制器 + RM_API2 | 本机 MotorBridge 串口直达六个 DM 电机 | 删除 IP/8080/RM_API 运行依赖，使用 `/dev/ttyACM0` 921600 |
| 失电保持 | 官方列出 J1-J3 机械制动、J4-J6 弹簧制动 | 目标硬件按失电直接失能处理 | 启动只能是桌面支撑姿态；禁止未知姿态自动回零 |
| 负载 | 5 kg | 1.5 kg | 不能沿用 RM65 的负载、速度和大范围笛卡尔参数 |
| 臂展 | 610/627 mm | 767 mm | 不按臂展比例放大视觉位移；首测仅 25/25/60 mm 包络 |
| 本体质量 | 约 7.2/7.3 kg | 约 4.5 kg | 低惯量不等于可忽略跌落风险；采用更小加速度和单步限幅 |
| 笛卡尔控制 | 控制器 `movel/movej_p/movep_canfd` | 本地 URDF + Pinocchio | 每个视觉目标先 IK，再转六轴 MIT 目标 |
| 重力补偿 | 封装在控制器/厂商运动链 | 主机实时计算 `g(q)` | 500 Hz MIT 目标 + 当前位置 `g(q)` 前馈 |
| 灵巧手链路 | 可复用 RM 工具 RS485/手接口 | 没有 RM 控制器工具口 | 只允许 BrainCo 独立 USB-RS485 或 CANFD |
| 单位 | RM 关节接口常用度，位姿由 SDK 管理 | SDK/Pinocchio 使用弧度和米 | 参数明确以 `_rad`、`_m`、`_nm` 命名 |

## 3. 两个源仓库中实际复用/替换的部分

### `reBotArm_SmolVLA`

调研得到的重力补偿核心是：读取六轴当前角度，使用
`reBotArm_control_py.dynamics`/Pinocchio 计算 `g(q)`，在 DM MIT 命令中同时发送
位置目标、零速度、位置/阻尼增益和重力前馈力矩。DM 关节与反馈 ID 是
`0x01..0x06` / `0x11..0x16`，J1-J3 为 4340P，J4-J6 为 4310，控制频率 500 Hz。

新后端保留该低层控制思想，但不复制 LeRobot 设备层；它直接复用 Seeed 官方 SDK
作为 submodule，并为 BrainCo 末端单独提供“六轴、无 Seeed 夹爪”的 YAML，防止
`enable_all()` 误使能不存在或已经拆除的 0x07 电机。

### `Jetson_Femto_Realman_Brainco_Demo_RTMPose_TensorRT`

保留：Orbbec RGB-D、RTMDet/RTMPose TensorRT、最近人员深度选择、人体/双手
关键点、手掌三维位移、五指弯曲标定、防抖、GUI、ROS 话题和 BrainCo 直连实现。

替换：`RM65PalmFollower` 的网络连接、RM 控制器笛卡尔命令、RM IK/关节扫描、
悬空 safe-home、J5 奇异点规则，以及 RM 工具口手部链路。活动入口现在构造
`B601PalmFollower`；旧 Realman 类只作为上游算法参考留在单文件内部且是可选导入，
没有 RM_API2 时也能解析/运行 B601 入口。

## 4. B601 启停状态机

```text
失能、桌面完整支撑
  -> 操作者显式确认
  -> 读取 6 轴有效反馈
  -> 检查静止 + 接近标定桌面关节值
  -> FK 记录真实 TCP/姿态
  -> MIT 模式
  -> 以真实关节角为首个目标 + g(q)
  -> 使能并启动 500 Hz 流
  -> 等待人员 + 人工按 S
  -> 仅允许先抬升，再允许 XY
  -> H/正常退出：minimum-jerk 回桌面并验证
  -> 失能、断开
```

失败均遵循 fail-closed：确认缺失、任一反馈缺失、运动未停止、桌面姿态偏差、IK
失败、关节限位或单步过大时，不发对应运动命令。停电和物理断线不经过状态机，
因此仍必须依赖桌面/机械支撑、急停布置和现场人员。

## 5. 真机必须重新确认的量

1. 六个电机方向、零位、反馈 ID 与 DM URDF 一致。
2. 空载和安装 BrainCo 后的 `g(q)` 符号、比例与各轴力矩上限；末端负载会改变模型。
3. 平行桌面姿态的六轴实测弧度值及合理容差。
4. 机器人基座 Z 轴确实垂直离开桌面；否则必须先改坐标映射和桌面包络。
5. `camera_to_arm_matrix` 通过相机-机器人标定确认，不能直接假设原 RM65 外参仍成立。
6. `H` 回位全路径无碰撞；正常退出能到位，但断电测试只能在可靠物理支撑下进行。
