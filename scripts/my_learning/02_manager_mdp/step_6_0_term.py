"""
Step 6.0: 终止条件管理器 (TerminationManager)
================================================================
文件描述：完善 RL 闭环——加入终止条件，实现"倒地/出界/超时 → 自动重置"。

剥洋葱重点：
1. terminated（倒地/出界）vs truncated（超时）：PPO 对两种结束的处理不同。
2. time_out=True 标记超时终止，让值函数做正确的 bootstrap 估计。
3. RL 闭环 = 观测 → 动作 → 奖励 → 终止检查 → 重置，本文件首次打通全链路。

学习成果：
  运行后给大随机动作，终端将打印频繁的"触发终止"提示，环境自动重置。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/02_manager_mdp/step_6_0_term.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import math
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

# ==================== 1. 自定义终止逻辑 (核心新增) ====================

def my_term_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """判定：杆子角度是否过大"""
    asset = env.scene[asset_cfg.name]
    # 杆子角度在第 2 个关节 (index=1)
    pole_angle = torch.abs(asset.data.joint_pos[:, 1])
    # 返回一个布尔张量 [num_envs]，True 表示触发终止
    return pole_angle > threshold

def my_term_out_of_bounds(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, distance: float) -> torch.Tensor:
    """判定：小车是否跑太远"""
    asset = env.scene[asset_cfg.name]
    # 小车位置在第 1 个关节 (index=0)
    cart_pos = torch.abs(asset.data.joint_pos[:, 0])
    return cart_pos > distance

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
            "position_range": (-0.2, 0.2), # 缩短随机范围，方便观察
            "velocity_range": (-0.1, 0.1),
        },
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=lambda env: torch.ones(env.num_envs, device=env.device), weight=1.0)
    upright = RewTerm(
        func=lambda env, asset_cfg: torch.cos(env.scene[asset_cfg.name].data.joint_pos[:, 1]), 
        weight=5.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

# 完善：终止条件配置
@configclass
class TerminationsCfg:
    """定义何时该‘杀掉’这个环境并重置"""
    # (1) 倒地终止：超过 45 度 (pi/4)
    too_tilt = DoneTerm(
        func=my_term_angle, 
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": math.pi / 4.0}
    )
    # (2) 出界终止：超过 2.5 米
    out_of_track = DoneTerm(
        func=my_term_out_of_bounds, 
        params={"asset_cfg": SceneEntityCfg("robot"), "distance": 2.5}
    )
    # (3) 时间截断：5 秒上限
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(num_envs=4, env_spacing=5.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg() # 激活终止管理器

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0
        self.episode_length_s = 5.0 

# ==================== 3. 运行区 ====================

def main():
    env = ManagerBasedRLEnv(cfg=CartpoleEnvCfg())
    env.reset()
    
    print("[INFO] 完整闭环环境已就绪。观察自动重置现象...")

    step_count = 0
    while simulation_app.is_running():
        # 给比较大的随机动作，让它很容易倒下或跑飞
        actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2.0
        
        obs, rew, terminated, truncated, info = env.step(actions)

        # 自动重置逻辑：在 RL 训练循环中，这是核心
        # 只要 terminated (挂了) 或 truncated (到时了)，就触发重置
        dones = terminated | truncated
        if dones.any():
            # 提取出需要重置的环境 ID
            env_ids = dones.nonzero(as_tuple=False).flatten()
            print(f" >>> 环境 {env_ids.tolist()} 触发终止！正在重置...")
            env.reset(env_ids=env_ids)

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()