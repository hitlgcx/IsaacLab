# Isaac Lab 剥洋葱学习系列

一套以"最小可运行代码"为原则的 Isaac Lab 渐进式学习教程。
每个文件只新增**一个**知识点，在上一文件的基础上增量构建，最终完成 Go2 机器狗平地行走的完整训练。

---

## 目录结构

```
scripts/my_learning/
├── 01_sim_basics/         # 仿真引擎基础（场景搭建、几何体、关节体）
├── 02_manager_mdp/        # Manager-Based MDP 五大管理器
├── 03_skrl_training/      # skrl 训练框架（包装、算法、训练、推理、规模化）
├── 04_go2_locomotion/     # Go2 四足步态全流程（共 16 个步骤）
└── 05_advanced/           # 高级专题（DirectRL、相机、操作臂）[占位待实现]
```

---

## 环境准备

```bash
conda activate env_isaaclab
```

step_1_0 ~ step_19_0 均在此环境下直接用 `python` 运行。  
只有 `step_1_1_scene_cli.py` 使用 `./isaaclab.sh -p` 运行（演示 CLI 参数模式）。

---

## 两种运行模式

| 模式 | 示例 | 参数传递 |
|------|------|---------|
| **直接 python**（本系列默认） | `python scripts/my_learning/.../step_N_M.py` | 硬编码 |
| **./isaaclab.sh**（step_1_1 演示）| `./isaaclab.sh -p scripts/my_learning/.../step_1_1_scene_cli.py [--headless]` | 命令行动态传入 |

---

## 文件速览

### 01_sim_basics — 仿真引擎基础

| 文件 | 状态 | 核心知识点 |
|------|------|-----------|
| `step_1_0_scene.py` | 完成 | @configclass / InteractiveScene / sim.step() |
| `step_1_1_scene_cli.py` | 完成 | argparse / AppLauncher.add_app_launcher_args |
| `step_1_2_spawn_prims.py` | [占位] | 几何图元生成（盒/球/圆柱），底层 USD prim API |
| `step_1_3_rigid_body.py` | [占位] | RigidObject 物理控制，位置/速度读取 |
| `step_1_4_articulation.py` | [占位] | 独立关节体控制，joint_pos 读写循环 |

### 02_manager_mdp — MDP 五大管理器

| 文件 | 状态 | 核心知识点 |
|------|------|-----------|
| `step_2_0_obs.py` | 完成 | ObsTerm / ObsGroup / ManagerBasedEnv |
| `step_3_0_act.py` | 完成 | JointEffortActionCfg / scale 参数 |
| `step_4_0_event.py` | 完成 | EventTerm / mode="reset" / write_joint_state_to_sim |
| `step_5_0_reward.py` | 完成 | RewTerm / ManagerBasedRLEnv / 五元组返回值 |
| `step_6_0_term.py` | 完成 | TerminationTermCfg / terminated vs truncated |

### 03_skrl_training — skrl 训练框架

| 文件 | 状态 | 核心知识点 |
|------|------|-----------|
| `step_7_0_wrapper.py` | 完成 | SkrlVecEnvWrapper / skrl VecEnv 接口 |
| `step_8_0_algo_cfg.py` | 完成 | GaussianMixin / DeterministicMixin / PPO_CFG |
| `step_9_0_train.py` | 完成 | skrl Runner / headless 训练 |
| `step_10_0_play.py` | 完成 | agent.load() / enable_training_mode(False) |
| `step_11_0_scale.py` | 完成 | num_envs=4096 / mini_batches 调整 |

### 04_go2_locomotion — Go2 四足步态

| 文件 | 状态 | 核心知识点 |
|------|------|-----------|
| `step_12_0_import_go2.py` | 完成 | UNITREE_GO2_CFG / 12 个驱动关节 |
| `step_12_1_add_new_robot.py` | [占位] | 从 USD 创建自定义 ArticulationCfg |
| `step_13_0_action_pd.py` | 完成 | JointPositionActionCfg / use_default_offset |
| `step_14_0_domain_rand.py` | 完成 | mode="interval" / randomize_rigid_body_material |
| `step_15_0_height_scan.py` | 完成 | RayCasterCfg / GridPatternCfg / 133 维观测 |
| `step_15_1_contact_sensor.py` | [占位] | ContactSensorCfg / 足部接触力检测 |
| `step_15_2_frame_transformer.py` | [占位] | FrameTransformerCfg / 足部世界坐标追踪 |
| `step_16_0_command.py` | 完成 | UniformVelocityCommandCfg / generated_commands |
| `step_17_0_terrain.py` | 完成 | TerrainImporterCfg / 程序化地形矩阵（无课程）|
| `step_17_1_curriculum.py` | 完成 | CurriculumTermCfg / terrain_levels_vel 动态调度 |
| `step_18_0_train_go2_minimal.py` | 完成 | **2 条奖励 / 30 维观测** / 验证最小训练闭环 |
| `step_18_1_train_go2_full.py` | 完成 | **9 条奖励 / 48 维观测** / RunningStandardScaler |
| `step_19_0_play_go2.py` | 完成 | 推理对齐 / mean_actions / 实时速度同步 |

### 05_advanced — 高级专题（均为占位）

| 文件 | 参考官方教程 | 核心知识点 |
|------|------------|-----------|
| `step_20_0_direct_rl.py` | 03_envs/ | DirectRLEnv 对比 Manager-Based |
| `step_21_0_camera_raycaster.py` | 04_sensors/ | RayCast 相机，深度图观测 |
| `step_22_0_camera_usd.py` | 04_sensors/ | USD 相机，RGB 图像观测 |
| `step_23_0_diff_ik.py` | 05_controllers/ | 微分逆运动学（Franka 操作臂）|
| `step_24_0_osc.py` | 05_controllers/ | 操作空间力/位混合控制 |

---

## 学习路径

```
01_sim_basics/
  step_1_0 → step_1_1              # 场景基础 + CLI 模式对比
       ↓
02_manager_mdp/
  step_2_0 → step_3_0 → step_4_0  # 观测 → 动作 → 事件
  step_5_0 → step_6_0              # 奖励 → 终止条件
       ↓
03_skrl_training/
  step_7_0 → step_8_0 → step_9_0  # 包装器 → 算法 → 训练（CartPole 完整闭环）
  step_10_0 → step_11_0            # 推理 → 规模化
       ↓
04_go2_locomotion/
  step_12_0 → step_13_0            # 导入 Go2 → PD 控制
  step_14_0                        # 领域随机化
  step_15_0                        # 高度图传感器
  step_16_0                        # 速度指令管理器
  step_17_0 → step_17_1            # 程序化地形 → 课程学习
  step_18_0                        # ★ 最简 Go2 RL 训练（2 条奖励，约 25 分钟验证）
  step_18_1                        # ★ 完整 Go2 RL 训练（9 条奖励，约 12 小时）
  step_19_0                        # Go2 推理展示
```

> **建议**：先跑 step_18_0 验证训练能收敛，再跑 step_18_1 体会每个优化点的收益。

---

## 快速开始

```bash
conda activate env_isaaclab

# 最基础的场景（5 秒出现窗口）
python scripts/my_learning/01_sim_basics/step_1_0_scene.py

# CartPole 完整训练（约 30 秒）
python scripts/my_learning/03_skrl_training/step_9_0_train.py

# CartPole 推理（需先完成 step_9_0 训练）
python scripts/my_learning/03_skrl_training/step_10_0_play.py

# Go2 最简验证训练（约 25 分钟）
python scripts/my_learning/04_go2_locomotion/step_18_0_train_go2_minimal.py

# Go2 完整训练（约 12 小时，需先完成调试）
python scripts/my_learning/04_go2_locomotion/step_18_1_train_go2_full.py

# Go2 推理展示（需先完成 step_18_1）
python scripts/my_learning/04_go2_locomotion/step_19_0_play_go2.py
```

---

## 训练时长参考（RTX 5070 12GB，4096 envs）

| 任务 | 步数 | 预计时长 |
|------|------|---------|
| CartPole（step_9_0） | 2,400 | ~30 秒 |
| Go2 最简（step_18_0，2048 envs） | 1,000,000 | ~25 分钟 |
| Go2 完整快速验证（step_18_1） | 2,500,000 | ~12 小时 |
| Go2 完整高质量 | 50,000,000 | ~10 天 |

---

## 日志查看

```bash
conda activate env_isaaclab
tensorboard --logdir scripts/my_learning/logs/skrl
```
