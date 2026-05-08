"""
Step 28.0: CartPole RGB 训练 —— 视觉 RL 完整训练闭环
================================================================
文件描述：整合 step_25~27 的所有配置，完成 CartPole 视觉 RL 的完整训练。
          重点对比状态 RL（step_9_0）与视觉 RL 的配置差异和收敛速度差异。

剥洋葱重点（相比 step_27_0 的差异）：
1. timesteps 从 100 增加到 100,000，真正观察收敛过程。
2. num_envs 从 64 增加到 512（TiledCamera 平铺渲染在此数量效率最优）。
3. rollouts=64，mini_batches=8（适配 512 envs 的 batch size）。

对比 step_9_0（状态 RL，CartPole）：
  step_9_0：  观测 4 维状态 / MLP(64,64) / 4096 envs / ~30 秒收敛
  step_28_0： 观测 100×100 RGB / CNN+MLP / 512 envs / 预计 ~10-30 分钟收敛
  → 视觉 RL 收敛慢的原因：
    (1) 像素包含大量冗余信息，CNN 需要学习提取有用特征
    (2) 512 envs 比 4096 envs 样本效率低
    (3) CNN 参数量比 MLP 多，更新更慢

日志目录：my_learning/logs/skrl/cartpole_rgb
运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_28_0_train_cartpole_rgb.py

查看训练曲线：
  tensorboard --logdir scripts/my_learning/logs/skrl/cartpole_rgb
"""

import os
import math

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ==========================================
# 0. 启动引擎
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_tasks.manager_based.classic.cartpole.mdp import joint_pos_target_l2
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


# ==========================================
# 1. 场景配置
# ==========================================

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100, height=100,
    )


# ==========================================
# 2. MDP 配置（与官方 CartpoleEnvCfg 相同的奖励函数）
# ==========================================

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )

@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": True},
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)},
    )

@configclass
class RewardCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=joint_pos_target_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )

@configclass
class TermCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleRGBEnvCfg(ManagerBasedRLEnvCfg):
    # 512 envs：TiledCamera 平铺渲染在此数量效率最优
    # env_spacing=20：防止相邻环境出现在相机视野中
    scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TermCfg = TermCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# ==========================================
# 3. 主训练函数
# ==========================================

def main():
    print("\n[INFO] 初始化 512 个 CartPole 并行环境（视觉 RL）...")
    env = ManagerBasedRLEnv(cfg=CartpoleRGBEnvCfg())
    obs_shape = env.observation_manager.group_obs_dim["policy"]
    print(f"[INFO] 观测 shape: {obs_shape}  （对比 step_9_0 的 (4,)）")
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

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
                    {
                        "name": "features_extractor",
                        "input": "permute(OBSERVATIONS, (0, 3, 1, 2))",
                        "layers": [
                            {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                            "flatten",
                        ],
                        "activations": "relu",
                    },
                    {"name": "net", "input": "features_extractor", "layers": [512], "activations": "elu"},
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {
                        "name": "features_extractor",
                        "input": "permute(OBSERVATIONS, (0, 3, 1, 2))",
                        "layers": [
                            {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                            "flatten",
                        ],
                        "activations": "relu",
                    },
                    {"name": "net", "input": "features_extractor", "layers": [512], "activations": "elu"},
                ],
                "output": "ONE",
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 64,      # 每轮收集 64×512 = 32768 样本
            "learning_epochs": 4,
            "mini_batches": 8,   # 每次更新 mini-batch 大小 = 32768/8 = 4096
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-4,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "state_preprocessor": None,
            "state_preprocessor_kwargs": None,
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": None,
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.0,
            "value_loss_scale": 1.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
            "experiment": {
                "directory": os.path.join(LOGS_DIR, "skrl/cartpole_rgb"),
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            # 100,000 步：足以观察到收敛趋势（约 10-30 分钟，取决于 GPU）
            # 若想更充分收敛可改到 500,000（约 1 小时）
            "timesteps": 100_000,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    print("[INFO] 启动 CartPole RGB 视觉 RL 训练（100,000 步）...")
    print(f"[INFO] 日志目录: {os.path.join(LOGS_DIR, 'skrl/cartpole_rgb')}")
    print("[INFO] 对比 step_9_0：状态 RL 约 30 秒收敛，视觉 RL 预计 10-30 分钟")
    runner = Runner(env, experiment_cfg)
    runner.run()

    print("\n[SUCCESS] 训练完成！用 tensorboard 查看 reward 曲线，对比 step_9_0 的收敛速度。")
    print(f"          tensorboard --logdir {os.path.join(LOGS_DIR, 'skrl')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
