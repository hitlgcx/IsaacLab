"""
Step 2.0: 观测管理器 (ObservationManager)
================================================================
文件描述：在 Step 1 场景基础上，接入观测管理器，让环境能够感知机器人状态。

剥洋葱重点：
1. 自定义观测函数只需返回一个 Tensor，框架自动完成批量化和拼接。
2. ObsTerm 是函数 + 参数的绑定容器，ObsGroup 是多个 ObsTerm 的集合。
3. ManagerBasedEnv 用于纯控制场景；RL 场景需升级为 ManagerBasedRLEnv（Step 5 引入）。
4. ActionsCfg 此时只是占位符：即便不控制，ManagerBasedEnv 也需要定义动作。

学习成果：
  运行后终端每 50 步打印一次观测张量，形状为 [2, 4]（2 个环境，4 个关节值）。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/02_manager_mdp/step_2_obs.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==================== 1. 自定义观测逻辑 ====================

def my_custom_get_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """直接从物理对象中提取关节位置张量"""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos

def my_custom_get_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """提取关节速度张量"""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel

# ==================== 2. 图纸设计 (配置区) ====================

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
        joint_vel = ObsTerm(func=my_custom_get_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# 新增：最小动作配置（占位符）
@configclass
class ActionsCfg:
    """即便不控制，也必须定义的动作管理器配置"""
    # 我们定义一个对关节施加力的动作，但后面我们可以不给它输入有效值
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=1.0
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """总装配置：遵循官方标准的 post_init 风格"""
    # 1. 定义组件（类属性）
    scene = CartpoleSceneCfg(num_envs=2, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # 2. 定义运行参数（逻辑处理）
    def __post_init__(self):
        """在这里统一管理环境的频率和步长"""
        # 控制频率：物理引擎跑 2 步，算法交互 1 步
        self.decimation = 2
        # 仿真步长：120Hz
        self.sim.dt = 1 / 120.0
        # 渲染间隔：通常设为与 decimation 一致，保证视觉刷新同步
        self.sim.render_interval = self.decimation

# ==================== 3. 运行与验证 ====================

def main():
    print("[INFO] 正在初始化环境总管...")
    env = ManagerBasedEnv(cfg=CartpoleEnvCfg())
    
    obs, _ = env.reset()
    print("[INFO] 环境构建成功。开始每隔 50 步打印一次观测数据...")

    step_count = 0
    while simulation_app.is_running():
        # 既然有了 ActionManager，我们需要根据它的维度给出一个全 0 动作
        # 这样机器人就不会乱动，方便我们观察重力下的自然跌落数据
        empty_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        
        # 步进环境
        obs, _ = env.step(empty_actions)

        if step_count % 50 == 0:
            print(f"--- Step {step_count} ---")
            policy_data = obs["policy"]
            print(f"观测张量 [Batch, Dim] = {list(policy_data.shape)}:\n{policy_data}\n")

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()