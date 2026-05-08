"""
Step 13.0: 多维动作空间与 PD 控制
================================================================
文件描述：将 Go2 接入 RL 环境，验证 12 维关节位置控制和五元组解包协议。

剥洋葱重点：
1. ManagerBasedRLEnv.step() 返回五元组（obs, rew, terminated, truncated, extras），不能少解包。
2. JointPositionActionCfg 使用底层 PD 控制器跟踪目标关节角度，而非直接施加力矩。
3. use_default_offset=True：网络输出 0 时，关节保持默认站立姿态而非零角度。

学习成果：
  运行后你将看到 Go2 在 PD 控制下保持近似站立姿态，终端实时打印前三个关节角度。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_13_0_action_pd.py
"""

# 0. 启动仿真引擎
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
    """定义动作：12 维关节位置控制"""
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
class RewardsCfg: pass
@configclass
class TerminationsCfg: pass
@configclass
class EventCfg: pass

# ==================== 2. 环境总装 ====================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=1, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4  # 50Hz
        self.episode_length_s = 20.0

def main():
    # 实例化环境
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    
    # 获取动作维度
    action_dim = env.action_manager.total_action_dim
    
    print(f"--- 剥洋葱 Step 13 验证 ---")
    print(f"环境已就绪！")
    print(f"动作空间维度: {action_dim}") 
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 1. 产生全 0 动作（维持默认站姿）
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            
            # 2. 核心修正：解包 5 个值 (obs, rew, terminated, truncated, extras)
            obs_dict, reward, terminated, truncated, extras = env.step(actions)
            
            # 3. 提取观测并验证
            current_obs = obs_dict["policy"]
            # 实时打印：你可以看到关节角度在重力作用下微小抖动
            print(f"实时关节角度 (前三个): {current_obs[0, :3].cpu().numpy()}", end='\r')
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()