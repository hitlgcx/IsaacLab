"""
Step 15.0: 感知层——引入高度图扫描 (Height Scan)
================================================================
文件描述：为 Go2 安装"脚下探测雷达"，让策略能感知地形起伏，为避障训练打基础。

剥洋葱重点：
1. RayCasterCfg 在机器人躯干正下方发射 11×11 = 121 条垂直射线，测量地面高度。
2. offset=0.5 将探测高度归一化：值为 0 表示地面与足底齐平。
3. 观测拼接：12（关节）+ 121（高度点）= 133 维，concatenate_terms=True 自动完成。

学习成果：
  运行后终端实时打印足底探测平均高度（约 0.0m 表示传感器工作正常）。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_15_0_height_scan.py
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
# 核心修正：严格按照最新版路径导入
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg 
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
    
    # 【剥洋葱核心】：高度扫描传感器
    # 修正点：移除了失效的 attach_to_visual_mesh 参数
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", # 挂载在躯干
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), # 高空俯射
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.0, 1.0)), # 11x11 网格
        mesh_prim_paths=["/World/defaultGroundPlane"], # 探测目标
        debug_vis=True, # 开启后能看到红色点阵
    )

@configclass
class Go2ActionCfg:
    """延续 Step 13 肌肉控制"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True 
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # (1) 基础：12 关节
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # (2) 感知：高度扫描 (121 点)
        # offset=0.5 用于将探测到的数值归一化到足底水平面附近
        height_scan = ObsTerm(
            func=mdp.height_scan, 
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5} 
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True # 洋葱拼接：12 + 121 = 133 维
    policy: PolicyCfg = PolicyCfg()

@configclass 
class EventCfg: pass
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
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4 
        self.episode_length_s = 20.0

def main():
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    
    # 自动获取总维度，验证拼接逻辑
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    
    print(f"--- 剥洋葱 Step 15 验证 ---")
    print(f"感知层开启成功！当前观测总维度: {obs_dim}") 
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            # 严格 5 值协议
            obs_dict, _, _, _, _ = env.step(actions)
            
            # 实时数据：打印第 12 位后的探测高度均值
            h_data = obs_dict["policy"][0, 12:]
            print(f"足底探测平均高度: {h_data.mean().item():.4f} m", end='\r')
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()