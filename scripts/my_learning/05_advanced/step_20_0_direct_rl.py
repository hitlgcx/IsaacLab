"""
Step 20.0: DirectRLEnv 模式——直接子类化 (Direct RL)
================================================================
文件描述：使用 DirectRLEnv 而非 ManagerBasedRLEnv 实现 CartPole 环境，
          对比两种开发模式的代码量、灵活性和适用场景。

剥洋葱重点：
1. DirectRLEnv 子类需手动实现 _get_observations / _get_rewards / _get_dones。
2. 不使用 Manager 层：观测、奖励、终止条件全部写在同一个类里，适合快速原型。
3. 与 Manager-Based 的权衡：DirectRL 更灵活，但复用性差；Manager-Based 模块化强。

学习成果：
  运行后 CartPole 环境正常运转，理解两种模式在同一任务上的代码差异。

参考官方教程：scripts/tutorials/03_envs/create_cartpole_base_env.py（DirectRL 版本）

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/05_advanced/step_20_direct_rl.py

待实现：参照官方 Direct RL 环境示例完成本文件。
"""

# TODO: 参照 source/isaaclab_tasks/direct/ 下的示例实现 DirectRLEnv 子类
pass
