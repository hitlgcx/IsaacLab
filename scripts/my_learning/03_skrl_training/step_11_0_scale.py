"""
Step 11.0: 规模化训练 (4096 并行环境)
================================================================
文件描述：将 Step 9 的 64 个环境扩展到 4096 个，体验 GPU 并行加速的巨大威力。

剥洋葱重点：
1. num_envs 从 64 → 4096：单次更新数据量从 1024 条激增到 65536 条。
2. mini_batches 需同步增大（4→32），防止 65536 条数据一次性撑爆显存。
3. 吞吐量提升的上限由 GPU 物理步推理的串行依赖决定，不能无限线性扩展。

学习成果：
  运行后以 4096 个并行环境训练 100 万步，约十几秒完成，终端显示极高的 it/s。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/03_skrl_training/step_11_0_scale.py
"""

# ==========================================
# 0. 启动引擎（无头模式，GPU 资源全部用于物理计算）
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

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

# ==========================================
# 1. 环境图纸：召唤千军万马！
# ==========================================
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())
    
    # 4096 并行环境（RTX 5070 12G 约占用 38% 显存）
    scene = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    
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
# 2. 核心：处理海量数据的训练引擎
# ==========================================
def main():
    print(f"\n[INFO] 1. 正在向 GPU 部署 4096 个并行物理环境...请稍候...")
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg = {
        "seed": 42,
        "models": {
            "separate": False,
            "policy": {"class": "GaussianMixin", "clip_actions": False, "clip_log_std": True, "min_log_std": -20.0, "max_log_std": 2.0, "initial_log_std": 0.0, "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}], "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "clip_actions": False, "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}], "output": "ONE"}
        },
        "memory": {
            "class": "RandomMemory",
            "memory_size": -1, 
        },
        "agent": {
            "class": "PPO",
            "rollouts": 16,       # 每环境 16 步 → 总计 4096×16 = 65536 条经验
            "learning_epochs": 4,
            "mini_batches": 32,   # 切 32 份防止 65536 条数据撑爆显存
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 5.0e-04,
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
                "directory": "logs/skrl/cartpole_massive",
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": 200,
            }
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 1000000,  # 4096 并行下约十几秒完成
            "environment_info": "log",
            "close_environment_at_exit": False
        }
    }

    print("[INFO] 2. 环境部署完毕！开始训练...")
    print(f"       -> 单次更新收集数据量: 4096 envs × 16 steps = {4096 * 16} 条经验")

    runner = Runner(env, experiment_cfg)
    runner.run()

    print("\n[INFO] 100 万步训练完成！模型已保存至 logs/skrl/cartpole_massive 目录。")

if __name__ == "__main__":
    main()
    simulation_app.close()