"""
Step 5.0: 奖励管理器 (RewardManager)
================================================================
文件描述：升级到 ManagerBasedRLEnv，接入奖励管理器，获得真正的 RL 训练信号。

剥洋葱重点：
1. ManagerBasedRLEnv（vs ManagerBasedEnv）：step() 返回五元组 (obs, rew, terminated, truncated, info)。
2. RewTerm 将多个奖励函数加权求和：总奖励 = Σ(weight_i × func_i())。
3. TerminationsCfg 是 RL 环境的必须占位，即使只写 time_out 也不能省略。

学习成果：
  运行后终端每 50 步打印当前总奖励值（存活 +1，杆子直立 +5cos(θ)）。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/02_manager_mdp/step_5_0_reward.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm # 必须导入
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp 
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==================== 1. 自定义逻辑 ====================

def my_reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def my_reward_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pole_angle = asset.data.joint_pos[:, 1]
    return torch.cos(pole_angle)

# ==================== 2. 配置区 ====================

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.joint_pos, 
                            params={"asset_cfg": SceneEntityCfg("robot")})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.1, 0.1),
        },
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=my_reward_alive, weight=1.0)
    upright = RewTerm(func=my_reward_upright, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot")})

# 新增：即便在 Step 5，我们也必须先定义这个占位符
@configclass
class TerminationsCfg:
    """终止条件配置（暂时留空或只写超时）"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(num_envs=2, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg() # 补齐挂载

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0
        # 补齐 RL 强制要求的参数：单局最大时长（秒）
        self.episode_length_s = 5.0 

# ==================== 3. 运行区 ====================

def main():
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    env.reset()
    
    print("[INFO] 强化学习环境已就绪。观察每一步返回的 Reward...")

    step_count = 0
    while simulation_app.is_running():
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        obs, rew, terminated, truncated, info = env.step(actions)

        if step_count % 50 == 0:
            # 此时你会发现返回的 rew 是有值的！
            print(f"Step {step_count} | 当前总奖励: {rew[0].item():.4f}")

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()