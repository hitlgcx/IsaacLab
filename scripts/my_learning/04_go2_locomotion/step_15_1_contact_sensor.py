"""
Step 15.1: 接触传感器 (Contact Sensor)
================================================================
文件描述：在 Go2 全身配置 ContactSensorCfg，读取各链接的接触力，
          理解足部接触检测在步态奖励（feet_air_time）和终止条件中的作用。

剥洋葱重点：
1. ContactSensorCfg 的 prim_path 必须覆盖机器人全部链接（Go2 共 19 个），不能用子集。
2. filter_prim_paths_expr 过滤"与哪些外部物体接触"，不限制传感器自身范围。
3. sensor.data.net_forces_w 返回 (num_envs, num_bodies, 3) 的接触力张量。

学习成果：
  运行后终端实时打印各足部链接的法向接触力，站立时非零，腾空时接近零。

参考官方教程：scripts/tutorials/04_sensors/add_sensors_on_robot.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_15_1_contact_sensor.py

待实现：参照 scripts/tutorials/04_sensors/add_sensors_on_robot.py 并结合 step_15 完成本文件。
"""

# TODO: 参照 scripts/tutorials/04_sensors/add_sensors_on_robot.py 实现 ContactSensor 部分
pass
