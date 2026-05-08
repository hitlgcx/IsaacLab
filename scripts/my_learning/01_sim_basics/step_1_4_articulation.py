"""
Step 1.4: 独立关节体控制 (Articulation)
================================================================
文件描述：在最简场景中加载带关节的机器人，读取 joint_pos / joint_vel，
          发送位置/力矩指令，理解 Articulation 数据结构和 write → step → read 循环。

剥洋葱重点：
1. ArticulationCfg 描述 USD 路径、初始状态、执行器参数。
2. articulation.data.joint_pos 是 (num_envs, num_joints) 形状的 GPU 张量。
3. set_joint_position_target() 写入目标 → sim.step() 推进物理 → 读取新状态。

学习成果：
  运行后能在终端看到关节角度随时间变化，理解主仿真循环与资产 API 的配合。

参考官方教程：scripts/tutorials/01_assets/run_articulation.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/01_sim_basics/step_1_4_articulation.py

待实现：参照 scripts/tutorials/01_assets/run_articulation.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/01_assets/run_articulation.py 实现
pass
