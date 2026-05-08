"""
Step 29.0: CartPole RGB 推理 —— 加载 CNN 策略并可视化
================================================================
文件描述：加载 step_28_0 训练好的 CNN checkpoint，在可视化模式下运行推理。
          展示视觉 RL 策略的实际表现，对比状态 RL 推理（step_10_0）的代码差异。

剥洋葱重点：
1. 推理配置的三个关键操作（与 step_10_0_play.py 完全相同的模式）：
   ① runner.agent.load(path)               → 恢复网络权重
   ② enable_training_mode(False, ...)      → 关闭探索噪声，输出最确定动作
   ③ outputs[-1].get("mean_actions", ...)  → 取均值动作（GaussianMixin 确定性输出）
2. agent.act(obs, states, timestep, timesteps) 需要 states = env.state()
   （skrl 内部用 states 维护 RNN 状态等，CartPole 中为空但必须传入）
3. 与 step_10_0 对比：
   - 唯一差异：环境配置换成了图像观测版本（TiledCamera + mdp.image）
   - 推理逻辑：100% 相同，体现了 Manager-Based 接口的一致性

注意：
  运行前需先完成 step_28_0 的训练。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_29_0_play_cartpole_rgb.py
"""

import os
import glob
import math
import time

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ==========================================
# 0. 启动引擎（headless=False 用于可视化）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
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
# 1. 环境配置（必须与 step_28_0 完全一致）
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
    # 推理时 16 个环境足够，减少渲染压力
    scene: SceneCfg = SceneCfg(num_envs=16, env_spacing=20.0)
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
# 2. 找到最新的 checkpoint
# ==========================================

def find_latest_checkpoint(log_root: str) -> str:
    for pattern in [
        os.path.join(log_root, "skrl/cartpole_rgb", "*", "checkpoints", "best_agent.pt"),
        os.path.join(log_root, "skrl/cartpole_rgb", "*", "checkpoints", "agent_*.pt"),
    ]:
        files = glob.glob(pattern)
        if files:
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        f"未找到 checkpoint，请先运行 step_28_0_train_cartpole_rgb.py 完成训练。\n"
        f"搜索路径: {os.path.join(log_root, 'skrl/cartpole_rgb')}"
    )


# ==========================================
# 3. 主函数
# ==========================================

def main():
    checkpoint_path = find_latest_checkpoint(LOGS_DIR)
    print(f"\n[INFO] 加载 checkpoint: {checkpoint_path}")

    env = ManagerBasedRLEnv(cfg=CartpoleRGBEnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    dt = env.unwrapped.step_dt

    # 推理时网络结构必须与训练完全一致
    experiment_cfg = {
        "models": {
            "separate": False,
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False, "clip_log_std": True,
                "min_log_std": -20.0, "max_log_std": 2.0, "initial_log_std": 0.0,
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
            "rollouts": 64,
            "gae_lambda": 0.95,
            "discount_factor": 0.99,
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
                "write_interval": 0,      # 推理时不写日志
                "checkpoint_interval": 0, # 推理时不保存 checkpoint
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 0,               # 0 = 不训练，只加载和推理
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    runner = Runner(env, experiment_cfg)
    runner.agent.load(checkpoint_path)
    # ① 关闭探索噪声，策略输出最确定的动作
    runner.agent.enable_training_mode(False, apply_to_models=True)

    print("[INFO] 开始推理（按 Ctrl+C 退出）...")
    print("[INFO] 与 step_10_0_play.py 的对比：推理逻辑 100% 相同，只有环境配置不同")

    obs, _ = env.reset()
    states = env.state()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            # ② 取均值动作（GaussianMixin 的确定性输出，无采样噪声）
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rew, terminated, truncated, info = env.step(actions)
            states = env.state()

        # 实时同步（防止渲染太快）
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
