"""
Step 3.0: 动作管理器 (ActionManager)
================================================================
文件描述：在 Step 2 基础上，接入动作管理器，用正弦波信号驱动小车运动。

剥洋葱重点：
1. JointEffortActionCfg 将算法输出的原始值乘以 scale 后施加到关节力矩。
2. scale=100.0 意味着神经网络输出 1.0 时，实际对关节施加 100N 的力。
3. 动作张量形状必须是 [num_envs, action_dim]，框架负责广播到各个环境。

学习成果：
  运行后你将看到小车按正弦波规律左右摆动，终端打印动作值与观测位置的实时对比。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/02_manager_mdp/step_3_0_act.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# 引入官方 MDP 库中的动作配置
import isaaclab.envs.mdp as mdp 
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==================== 1. 沿用 Step 2 的观测逻辑 ====================

def my_custom_get_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos

# ==================== 2. 配置区 (图纸设计) ====================

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg())

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=my_custom_get_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """核心变化：正式配置动作管理器"""
    # 我们控制名为 "slider_to_cart" 的关节（也就是小车的底座）
    # 使用力矩控制 (Effort)，并设置缩放系数为 100.0
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=["slider_to_cart"], 
        scale=100.0
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    scene = CartpoleSceneCfg(num_envs=2, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation

# ==================== 3. 运行与控制验证 ====================

def main():
    env = ManagerBasedEnv(cfg=CartpoleEnvCfg())
    env.reset()
    
    print(f"[INFO] 动作维度: {env.action_manager.total_action_dim}")
    print("[INFO] 开始运行，小车将按照正弦波规律摆动...")

    step_count = 0
    while simulation_app.is_running():
        # --- 控制逻辑：生成一个正弦波动作 ---
        # 动作张量形状必须是 [num_envs, action_dim]
        # 这里的动作值在 -1.0 到 1.0 之间变化
        time_sec = step_count * env.step_dt
        sine_val = math.sin(time_sec * 2.0 * math.pi * 0.5) # 0.5Hz 的频率
        
        actions = torch.full((env.num_envs, env.action_manager.total_action_dim), sine_val, device=env.device)
        
        # 步进环境：ActionManager 会自动执行：sine_val * scale(100.0) -> 施加到物理引擎
        obs, _ = env.step(actions)

        if step_count % 50 == 0:
            # 观察观测到的位置数据是否随着动作在摆动
            print(f"Step {step_count} | 动作输入: {sine_val:.2f} | 观测位置: {obs['policy'][0, 0]:.4f}")

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()