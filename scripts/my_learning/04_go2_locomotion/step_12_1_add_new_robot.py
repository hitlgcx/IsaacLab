"""
Step 12.1: 从 USD 创建自定义机器人配置 (Add New Robot)
================================================================
文件描述：演示如何将任意 USD 机器人资产注册为 Isaac Lab 的 ArticulationCfg，
          并使其在 ManagerBasedRLEnv 中可用，理解自定义机器人接入的完整流程。

剥洋葱重点：
1. ArticulationCfg 的 spawn=UsdFileCfg(usd_path=...) 指向自定义 USD 文件。
2. init_state 中的 joint_pos 字典用关节名称（支持正则）配置初始角度。
3. actuators 字典将执行器模型（ImplicitActuator / DCMotor）绑定到关节组。

学习成果：
  运行后，自定义机器人出现在场景中并能接受关节指令，整个流程与 Go2 完全一致。

参考官方教程：scripts/tutorials/01_assets/add_new_robot.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_12_1_add_new_robot.py

待实现：参照 scripts/tutorials/01_assets/add_new_robot.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/01_assets/add_new_robot.py 实现
pass
