"""
Step 24.0: 操作空间控制 (Operational Space Control)
================================================================
文件描述：使用 OperationalSpaceController 实现末端执行器的力/位混合控制，
          理解 OSC 与纯位置 IK（step_23）在阻抗控制场景下的区别。

剥洋葱重点：
1. OSC 同时控制末端位置和接触力，适合需要柔顺交互的操作任务。
2. OperationalSpaceControllerCfg 的 motion_control_axes / contact_wrench_control_axes
   决定哪些自由度走位置控制，哪些走力控制。
3. 需要质量矩阵 (mass_matrix) 和雅可比矩阵输入，计算成本高于纯 IK。

学习成果：
  运行后 Franka 末端以可控力与桌面接触，终端打印接触力值，验证力控回路正常。

参考官方教程：scripts/tutorials/05_controllers/run_osc.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/05_advanced/step_24_osc.py

待实现：参照 scripts/tutorials/05_controllers/run_osc.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/05_controllers/run_osc.py 实现
pass
