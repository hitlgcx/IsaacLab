"""
Step 14.0: 领域随机化 (Domain Randomization)
================================================================
文件描述：引入 EventManager，对物理参数和外部扰动进行随机化，提升策略的泛化能力。

剥洋葱重点：
1. mode=”reset” 事件：每次环境重置时触发（如随机化地面摩擦力）。
2. mode=”interval” 事件：运行中按时间间隔随机触发（如模拟被踢扰动）。
3. 领域随机化是 sim-to-real 迁移的核心手段：让策略在随机物理世界中学会鲁棒行为。

学习成果：
  运行后观察机器狗是否会突然”抽动”（被随机施加速度脉冲），这是 DR 的直观体现。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_14_0_domain_rand.py
"""

# 0. 启动引擎
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 导入 Go2 配置
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# ==================== 1. 配置区 ====================

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

@configclass
class Go2ActionCfg:
    """延续 Step 13：维持站姿"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True 
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """【剥洋葱核心】：定义随机化事件"""
    
    # (1) 地面摩擦力随机化：在每次环境重置时触发
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64, # 分桶提升计算效率
        },
    )

    # (2) 外部推力扰动：在环境运行期间每隔一段时间触发（模拟被踢）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0), # 每 2-4 秒踢一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, # 给予水平方向随机速度
        },
    )

@configclass
class RewardsCfg: pass
@configclass
class TerminationsCfg: pass

# ==================== 2. 环境总装 ====================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=1, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()  # 挂载随机化事件
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4  # 50Hz
        self.episode_length_s = 20.0

def main():
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    action_dim = env.action_manager.total_action_dim
    
    print(f"--- 剥洋葱 Step 14 验证 ---")
    print(f"领域随机化已激活！请观察机器狗是否会突然“抽动”（被随机踢）。")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 持续发送全 0，观察机器狗在扰动下的恢复能力
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            obs, rew, terminated, truncated, extras = env.step(actions)
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()