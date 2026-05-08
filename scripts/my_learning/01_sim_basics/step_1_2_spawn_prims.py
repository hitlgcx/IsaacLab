"""
Step 1.2: 生成几何图元 (Spawn Primitives)
================================================================
文件描述：在 Isaac Sim 中直接生成基础几何体（立方体、球体、圆柱体），
          演示底层 USD prim 生成 API，理解 PhysicsScene 与 RigidBody 的关系。

剥洋葱重点：
1. spawn_utils 提供 CuboidCfg / SphereCfg / CylinderCfg 等配置类。
2. 每个 prim 可独立配置质量、摩擦力、碰撞体积。
3. 与 step_1 的区别：这里不使用 SceneBuilder，直接操作底层 prim API。

学习成果：
  运行后看到仿真场景中出现各种几何体从高处落下并与地面碰撞。

参考官方教程：scripts/tutorials/00_sim/spawn_prims.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/01_sim_basics/step_1_2_spawn_prims.py

待实现：参照 scripts/tutorials/00_sim/spawn_prims.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/00_sim/spawn_prims.py 实现
pass
