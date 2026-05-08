"""
Step 10.0: 推理展示 (CartPole)
================================================================
文件描述：加载 Step 9 训练好的模型，用图形界面展示策略的实际效果。

剥洋葱重点：
1. runner.agent.load(path) 从检查点文件恢复网络权重。
2. enable_training_mode(False) 切换为推理模式：关闭探索噪声，输出最确定的动作。
3. outputs[-1].get("mean_actions") 取均值动作，展示策略的最优表现。

学习成果：
  运行后 4 台小车展示已训练策略，杆子保持直立状态。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/03_skrl_training/step_10_0_play.py  （需先运行 step_9_train.py）

前置依赖：
  step_9_train.py（生成 logs/skrl/cartpole_minimal 模型文件）
"""

# ==========================================
# 0. 启动引擎（开启图形界面以展示推理结果）
# ==========================================
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import os
import glob
import sys
import time
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

# --- skrl 组件 ---
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

# ==========================================
# 1. 环境图纸 (专门为观赏优化)
# ==========================================
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())
    
    # 推理时只用 4 个环境，便于观察
    scene = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    
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
# 2. 自动寻找最新模型
# ==========================================
def get_latest_model_path():
    search_dir = "logs/skrl/cartpole_minimal"
    pt_files = glob.glob(f"{search_dir}/**/agent*.pt", recursive=True)
    if not pt_files:
        print(f"[ERROR] 在 {search_dir} 中没有找到任何模型文件！请确认 Step 9 训练成功。")
        sys.exit(1)
    latest_model = max(pt_files, key=os.path.getctime)
    return latest_model

# ==========================================
# 3. 核心：推理循环 (Inference Loop)
# ==========================================
def main():
    print("\n[INFO] 1. 正在初始化物理环境 (4个环境观赏模式)...")
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print("[INFO] 2. 正在加载推理配置（必须与训练配置结构完全一致）...")
    # 推理时必须提供完整的字典以通过 Runner 的结构验证
    experiment_cfg = {
        "models": {
            "separate": False,
            "policy": {"class": "GaussianMixin", "clip_actions": False, "clip_log_std": True, "min_log_std": -20.0, "max_log_std": 2.0, "initial_log_std": 0.0, "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}], "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "clip_actions": False, "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [32, 32], "activations": "elu"}], "output": "ONE"}
        },
        "memory": {
            "class": "RandomMemory",
            "memory_size": -1, # Runner 占位必须
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
                "write_interval": 0,       # 推理时不写日志
                "checkpoint_interval": 0,  # 推理时不保存模型
            }
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 0,            # 推理时直接设为 0
            "environment_info": "log",
            "close_environment_at_exit": False
        }
    }

    runner = Runner(env, experiment_cfg)

    model_path = get_latest_model_path()
    print(f"\n[INFO] 3. 找到并加载最新模型：{model_path}")
    runner.agent.load(model_path)
    
    # 彻底关闭探索噪声，让小车发挥真实水平
    runner.agent.enable_training_mode(False, apply_to_models=True)

    print("[INFO] 4. 开始播放！(按 Ctrl+C 退出)")
    
    # 获取环境步长，用于控制现实世界播放速度
    dt = env.unwrapped.step_dt
    
    obs, _ = env.reset()
    states = env.state()

    while simulation_app.is_running():
        start_time = time.time()

        # 使用 inference_mode 关闭梯度追踪
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            # 取均值动作（无噪声，展示最稳定效果）
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rew, terminated, truncated, info = env.step(actions)
            states = env.state()

        # 现实时间同步 (防止跑得太快)
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
    simulation_app.close()