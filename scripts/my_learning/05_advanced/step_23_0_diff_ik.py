"""
Step 23.0: 微分逆运动学 (Differential Inverse Kinematics)
================================================================
文件描述：为 Franka Emika 操作臂配置 DifferentialIKController，
          通过末端执行器目标位姿反解关节角度，理解操作任务的控制基础。

剥洋葱重点：
1. DifferentialIKControllerCfg 指定 command_type（position / pose）和 ik_method（dls / svd）。
2. 控制器输入末端目标位姿，输出 joint_pos 增量，由 ArticulationAction 施加。
3. 与 PD 关节控制（step_13）的区别：IK 在笛卡尔空间规划，适合操作任务。

学习成果：
  运行后 Franka 末端执行器跟踪指定轨迹，终端打印末端实际位置误差。

参考官方教程：scripts/tutorials/05_controllers/run_diff_ik.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/05_advanced/step_23_diff_ik.py

待实现：参照 scripts/tutorials/05_controllers/run_diff_ik.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/05_controllers/run_diff_ik.py 实现
pass
