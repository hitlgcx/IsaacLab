"""
Step 18.0: Go2 最简 RL 训练——打通第一个训练闭环
================================================================
文件描述：用最精简的配置（2 条奖励、30 维观测、无接触传感器、无归一化）
          打通 Go2 RL 训练的完整闭环，验证训练曲线能够上升。

剥洋葱重点：
1. 只用 2 条奖励就能开始训练：
     track_lin_vel_xy_exp  ← 主任务：跟上速度指令
     flat_orientation_l2   ← 约束：身体保持水平（防翻倒）
2. 30 维最小观测集：projected_gravity(3) + vel_cmd(3) + joint_pos_rel(12) + joint_vel_rel(12)
   有意省略 base_lin/ang_vel 和 actions，观察训练是否仍能收敛。
3. 不用 RunningStandardScaler，对比 step_18_1 体会归一化对收敛速度的影响。
4. scale=0.25 是 Go2 训练的关键参数，防止随机初始化时产生暴力动作导致出生即倒。

对比 step_18_1：
  本文件：2 条奖励 / 30 维观测 / 无归一化 → 验证最小可行性
  step_18_1：9 条奖励 / 48 维观测 / RunningStandardScaler → 生产质量

学习成果：
  训练约 100 万步后（RTX 5070 约 25 分钟），观察 reward_total 曲线是否上升。
  若曲线不上升，对比 step_18_1 的配置找出差距所在——这本身就是学习。
  日志保存至 my_learning/logs/skrl/go2_minimal，可用 tensorboard 查看。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_18_0_train_go2_minimal.py

后续：
  训练完成后可参考 step_19_0_play_go2.py 的结构自行编写对应的推理文件
  （注意：推理环境的观测配置必须与训练完全一致，否则 shape mismatch）。
  step_18_1_train_go2_full.py 展示加入全部奖励和优化技巧后的完整训练。
"""

import os

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ==========================================
# 0. 启动引擎（必须第一步）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

# 与 step_18_1 相同：使用本地 USD 路径（云端 S3 路径 404）
LOCAL_UNITREE_GO2_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(
        usd_path="/home/hit-lgc/HomeWorkspace/unitree_rl_lab/unitree_model/Go2/usd/go2.usd"
    )
)

# ==========================================
# 1. 自定义判定（不依赖接触传感器，用简单高度判定）
# ==========================================

def my_term_fallen(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """躯干 Z 高度 < 0.2m 即判定跌倒。比 illegal_contact 简单但触发稍晚。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < 0.2

# ==========================================
# 2. 场景配置（平面地形，不依赖课程）
# ==========================================

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    # 平面地形，摩擦力设为 1.0 防打滑（与 step_18_1 一致）
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = LOCAL_UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # 注意：无 ContactSensorCfg，所以不能使用 illegal_contact 和 feet_air_time

# ==========================================
# 3. 动作配置
# ==========================================

@configclass
class Go2ActionCfg:
    # scale=0.25：关键！防止随机初始化时产生暴力动作
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )

# ==========================================
# 4. 速度指令配置
# ==========================================

@configclass
class Go2CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,  # 10% 的环境保持零速，先学站稳再学走
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

# ==========================================
# 5. 观测配置（30 维最小集）
# ==========================================

@configclass
class Go2ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """
        30 维最小观测集（对比 step_18_1 的 48 维）：
          projected_gravity (3)  ← IMU 平衡感，关键的姿态感知
          velocity_command  (3)  ← 目标速度，策略需要知道"往哪走"
          joint_pos_rel     (12) ← 相对默认姿态的关节偏移，状态反馈
          joint_vel_rel     (12) ← 关节速度，动态感知
        有意省略：
          base_lin/ang_vel  (6)  ← 速度传感器（实物上不可靠，先验证不用也能跑）
          last_action       (12) ← 动作历史（先验证不用是否影响收敛）
        """
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        velocity_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# ==========================================
# 6. 奖励配置（只有 2 条）
# ==========================================

@configclass
class Go2RewardCfg:
    # 主任务：鼓励机器人跟上速度指令
    # std=0.5 比 step_18_1 的 sqrt(0.25) 更宽松，允许早期训练误差更大
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 约束：惩罚身体倾斜，防止机器人用"趴着走"的方式作弊
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

# ==========================================
# 7. 终止条件配置
# ==========================================

@configclass
class Go2TermCfg:
    # 简单高度判定（不需要接触传感器）
    base_fallen = DoneTerm(
        func=my_term_fallen, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # 时间截断：告知值函数"这是超时而非跌倒"，让 PPO 正确 bootstrap
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

# ==========================================
# 8. 重置事件配置（与 step_18_1 相同）
# ==========================================

@configclass
class Go2EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

# ==========================================
# 9. 总环境配置
# ==========================================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=2048, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: Go2ObsCfg = Go2ObsCfg()
    commands: Go2CommandCfg = Go2CommandCfg()
    rewards: Go2RewardCfg = Go2RewardCfg()
    terminations: Go2TermCfg = Go2TermCfg()
    events: Go2EventCfg = Go2EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

# ==========================================
# 10. 主训练函数
# ==========================================

def main():
    print("\n[INFO] 1. 初始化 2048 只 Go2 并行环境（最简配置）...")
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    print(f"[INFO] 观测维度: {obs_dim} 维（对比 step_18_1 的 48 维）")
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print("[INFO] 2. 加载 PPO 配置（无 RunningStandardScaler）...")
    experiment_cfg = {
        "seed": 42,
        "models": {
            "separate": False,
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "network": [
                    {"name": "net", "input": "OBSERVATIONS", "layers": [256, 128], "activations": "elu"}
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {"name": "net", "input": "OBSERVATIONS", "layers": [256, 128], "activations": "elu"}
                ],
                "output": "ONE",
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 24,
            "learning_epochs": 5,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-3,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            # 对比 step_18_1：不使用 RunningStandardScaler
            # 效果：训练可能更不稳定或收敛更慢，但逻辑更简单，便于理解基础流程
            "state_preprocessor": None,
            "state_preprocessor_kwargs": None,
            "value_preprocessor": None,
            "value_preprocessor_kwargs": None,
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 2.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": True,
            "experiment": {
                "directory": os.path.join(LOGS_DIR, "skrl/go2_minimal"),
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            # 1M 步 ≈ 25 分钟（RTX 5070，2048 envs）
            # 目标：验证曲线上升，不追求行走质量
            "timesteps": 1_000_000,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    print("[INFO] 3. 启动训练（目标：验证曲线上升，约 25 分钟）...")
    print(f"[INFO] 日志目录: {os.path.join(LOGS_DIR, 'skrl/go2_minimal')}")
    runner = Runner(env, experiment_cfg)
    runner.run()

    print(f"\n[SUCCESS] 完成！对比 step_18_1 的完整配置，观察收敛速度和最终性能的差距。")


if __name__ == "__main__":
    main()
    simulation_app.close()
