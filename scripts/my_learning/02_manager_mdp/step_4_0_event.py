"""
Step 4.0: 事件管理器 (EventManager)
================================================================
文件描述：在 Step 3 基础上，接入事件管理器，实现可控的随机重置。

剥洋葱重点：
1. EventTerm 绑定函数与触发模式：mode="reset" 在 env.reset() 时触发。
2. write_joint_state_to_sim() 是"瞬移"操作，直接写入物理引擎状态。
3. env_ids 参数让你只重置部分环境，其余环境继续运行不受影响。

学习成果：
  运行后每 200 步触发一次手动重置，可观察 0 号小车"瞬移"到随机位置，1 号不受影响。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/02_manager_mdp/step_4_0_event.py
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
from isaaclab.managers import EventTermCfg as EventTerm  # 新增导入
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# 引入动作占位逻辑
import isaaclab.envs.mdp as mdp 
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==================== 1. 自定义事件逻辑 (核心新增) ====================

def my_custom_reset_event(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """自定义重置函数：将机器人传送到随机位置"""
    # 获取机器人对象
    asset = env.scene[asset_cfg.name]
    
    # 1. 生成随机初始位置
    # 我们希望小车在 [-0.5, 0.5] 之间，杆子角度在 [-0.2, 0.2] 弧度之间
    # asset.data.joint_pos 形状是 [num_envs, num_joints]
    # 注意：我们只为 env_ids 指定的环境生成随机数
    num_resets = len(env_ids)
    rand_pos = (torch.rand((num_resets, asset.num_joints), device=env.device) - 0.5) * 1.0
    rand_vel = torch.zeros((num_resets, asset.num_joints), device=env.device)
    
    # 2. 核心操作：强制写入物理引擎
    # 这一步会直接“瞬移”机器人，不遵循物理运动规律
    asset.write_joint_state_to_sim(rand_pos, rand_vel, env_ids=env_ids)
    
    print(f" [EVENT] 已重置环境索引: {env_ids.tolist()}")

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
        joint_pos = ObsTerm(func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.joint_pos, 
                            params={"asset_cfg": SceneEntityCfg("robot")})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

# 新增：事件配置
@configclass
class EventCfg:
    """定义何时触发何种瞬间操作"""
    # mode="reset" 表示该函数在 env.reset() 被调用时运行
    reset_robot = EventTerm(
        func=my_custom_reset_event,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    scene = CartpoleSceneCfg(num_envs=2, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg() # 挂载事件管理器

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120.0

# ==================== 3. 运行与重置验证 ====================

def main():
    env = ManagerBasedEnv(cfg=CartpoleEnvCfg())
    
    # 第一次重置：触发所有 env_ids 的 reset event
    env.reset()
    
    step_count = 0
    while simulation_app.is_running():
        # 继续给正弦波动作，让小车动起来
        time_sec = step_count * env.step_dt
        sine_val = math.sin(time_sec * 2.0 * math.pi * 0.5)
        actions = torch.full((env.num_envs, env.action_manager.total_action_dim), sine_val, device=env.device)
        
        env.step(actions)

        # 每隔 200 步，我们手动触发一次重置，观察“瞬移”现象
        if step_count > 0 and step_count % 200 == 0:
            print(f"\n--- Step {step_count}: 触发手动重置 ---")
            # 我们可以选择只重置第 0 号环境，观察第 1 号环境是否受影响
            env_ids = torch.tensor([0], device=env.device)
            env.reset(env_ids=env_ids)

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()