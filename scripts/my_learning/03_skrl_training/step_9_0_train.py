"""
Step 9.0: 完整 PPO 训练流程 (CartPole)
================================================================
文件描述：整合前 8 步所有组件，用 skrl Runner 一行代码启动完整训练。

剥洋葱重点：
1. Runner 接管所有底层细节：自动建网络、建内存、建优化器、写日志。
2. headless=True 关闭渲染，GPU 资源全部用于物理计算，训练速度大幅提升。
3. experiment_cfg 字典即训练大纲，控制网络结构、超参数和日志行为。

学习成果：
  运行后开始训练 2400 步，日志保存到 logs/skrl/cartpole_minimal 目录。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/03_skrl_training/step_9_train.py
"""

# ==========================================
# 0. 启动引擎 (全局唯一，必须在最前)
# ==========================================
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils

# --- Isaac Lab 组件 ---
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm 
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp 
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# --- skrl 核心组件 ---
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

# ==========================================
# 1. 环境图纸 (保持不变)
# ==========================================
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())
    
    scene = CartpoleSceneCfg(num_envs=64, env_spacing=4.0)
    
    @configclass
    class ObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
            joint_vel = ObsTerm(func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        policy: PolicyCfg = PolicyCfg()
    observations = ObsCfg()

    @configclass
    class ActCfg:
        joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    actions = ActCfg()

    @configclass
    class EvtCfg:
        reset_robot = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)})
    events = EvtCfg()

    @configclass
    class RewCfg:
        alive = RewTerm(func=lambda env: torch.ones(env.num_envs, device=env.device), weight=1.0)
        upright = RewTerm(func=lambda env, asset_cfg: torch.cos(env.scene[asset_cfg.name].data.joint_pos[:, 1]), weight=5.0, params={"asset_cfg": SceneEntityCfg("robot")})
    rewards = RewCfg()

    @configclass
    class TermCfg:
        too_tilt = DoneTerm(func=lambda env, asset_cfg, threshold: torch.abs(env.scene[asset_cfg.name].data.joint_pos[:, 1]) > threshold, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.785})
        out_of_track = DoneTerm(func=lambda env, asset_cfg, distance: torch.abs(env.scene[asset_cfg.name].data.joint_pos[:, 0]) > distance, params={"asset_cfg": SceneEntityCfg("robot"), "distance": 2.5})
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terminations = TermCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0
        self.episode_length_s = 5.0 

# ==========================================
# 2. 核心：基于 Runner 的工业级训练
# ==========================================
def main():
    print("\n[INFO] 1. 正在初始化物理环境...")
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print("[INFO] 2. 正在加载训练配置大纲...")
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
                    {"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}
                ],
                "output": "ONE",
            }
        },
        "memory": {
            "class": "RandomMemory",
            "memory_size": -1, # 自动推断
        },
        "agent": {
            "class": "PPO",
            "rollouts": 16,
            "learning_epochs": 8,
            "mini_batches": 8,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3.0e-04,
            "learning_rate_scheduler": "KLAdaptiveLR", 
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
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
            "entropy_loss_scale": 0.0,
            "value_loss_scale": 2.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
            "experiment": {
                "directory": "logs/skrl/cartpole_minimal",
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            }
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 2400,
            "environment_info": "log",
            "close_environment_at_exit": False
        }
    }

    print("[INFO] 3. 正在启动 Runner 引擎进行训练...")
    runner = Runner(env, experiment_cfg)
    runner.run()
    
    print("\n[INFO] 训练完成！模型已保存至 logs/skrl/cartpole_minimal 目录。")

if __name__ == "__main__":
    main()
    simulation_app.close()