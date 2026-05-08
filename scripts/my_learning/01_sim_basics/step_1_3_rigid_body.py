"""
Step 1.3: 刚体物理控制 (Rigid Body)
================================================================
文件描述：创建并控制非关节刚体对象，读取位置/速度状态，施加外力，
          理解 RigidObject 与 Articulation 的根本区别。

剥洋葱重点：
1. RigidObjectCfg 配置质量、碰撞属性，不含关节自由度。
2. rigid_object.data.root_pos_w / root_vel_w 读取世界坐标状态。
3. 刚体只能通过施力/设速度控制，无法直接控制关节角度。

学习成果：
  运行后观察刚体在重力下的自由落体，并理解它与 Articulation（step_1_4）的边界。

参考官方教程：scripts/tutorials/01_assets/run_rigid_object.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/01_sim_basics/step_1_3_rigid_body.py

待实现：参照 scripts/tutorials/01_assets/run_rigid_object.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/01_assets/run_rigid_object.py 实现
pass
