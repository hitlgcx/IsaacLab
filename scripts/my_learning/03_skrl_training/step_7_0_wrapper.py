"""
Step 7.0: skrl 环境包装器 (SkrlVecEnvWrapper)
================================================================
文件描述：将 Isaac Lab 原生环境包装为 skrl 标准格式，打通框架对接的最后一公里。

剥洋葱重点：
1. SkrlVecEnvWrapper 将 Isaac Lab 环境转换为 skrl 识别的 VecEnv 接口。
2. 包装后 obs 从字典变为 Tensor，step() 接口与 skrl Agent 无缝对接。
3. 包装器本身不改变任何物理逻辑，只是做接口翻译。

学习成果：
  运行后终端打印包装前后的环境类型，以及包装后 obs 的张量形状。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/03_skrl_training/step_7_0_wrapper.py
"""

# ==========================================
# 1. 启动仿真应用 (必须在最前面，全局唯一)
# ==========================================
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})  # 开启图形界面以观察包装效果
simulation_app = app_launcher.app

import torch
import math
import gymnasium as gym
import isaaclab.sim as sim_utils
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

# 针对 skrl 的环境包装器
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ==========================================
# 2. 环境配置区 (从 Step 6 复制过来的精华)
# ==========================================

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(num_envs=4, env_spacing=5.0)
    
    @configclass
    class ObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
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
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terminations = TermCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0
        self.episode_length_s = 5.0 

# ==========================================
# 3. 包装器测试逻辑 (Step 7 核心)
# ==========================================

def main():
    print("\n" + "="*50)
    print("[INFO] 正在创建原生 Isaac Lab 环境...")
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    print(f"原生环境类型: {type(env)}")
    print("="*50 + "\n")

    print("\n" + "="*50)
    print("[INFO] 正在应用 SkrlVecEnvWrapper 换装...")
    # 这个包装器将环境转为 skrl 标准格式
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    print(f"包装后环境类型: {type(env)}")
    print("="*50 + "\n")

    # 包装后的 reset 返回值直接就是 Tensor，不再是字典
    obs, _ = env.reset()
    print(f"[验证] 包装后的 obs 形状: {obs.shape}")

    step_count = 0
    while simulation_app.is_running():
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        # 包装后的 step 接口与 skrl 无缝对接
        obs, rew, terminated, truncated, info = env.step(actions)

        if step_count % 50 == 0:
            print(f"Step {step_count} | Reward Shape: {rew.shape} | Obs Shape: {obs.shape}")

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()