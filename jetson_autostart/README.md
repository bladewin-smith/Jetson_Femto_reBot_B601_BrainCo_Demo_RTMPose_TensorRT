# Jetson 启动脚本

1. 复制 `b601_brainco_orbbec_demo.env.example` 为
   `b601_brainco_orbbec_demo.env`。
2. 先保持 `ROBOT_MOTION_ENABLED=false` 完成纯视觉和灵巧手测试。
3. 用 `scripts/read_b601_table_pose.py` 读取桌面支撑姿态，写入
   `B601_TABLETOP_JOINTS_RAD`。
4. 机械臂断电、平行且完整支撑在桌面上，清空工作区；随后才把
   `ROBOT_MOTION_ENABLED` 与 `B601_TABLETOP_STARTUP_CONFIRMED` 同时设为 `true`。
5. 运行 `bash jetson_autostart/run_b601_brainco_orbbec_demo.sh`，识别稳定后按
   `S` 启动，按 `H` 缓慢回桌面姿态并暂停。
