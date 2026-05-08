"""
Step 16.0: 指令管理器 (Command Manager)
================================================================
文件描述：引入速度指令生成器，让策略学会"听指令行动"而非无目的运动。

剥洋葱重点：
1. CommandsCfg 作为独立模块生成目标速度指令（vx, vy, ωz），每隔几秒重新采样。
2. 指令必须通过 mdp.generated_commands 注入到观测中，策略才能"看到"目标。
3. 解耦设计：指令生成器、观测、奖励各自独立——通过 "base_velocity" 名称字符串关联。

学习成果：
  运行后终端实时打印当前目标速度指令，每 3 秒自动更换一次。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_16_0_command.py
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
    
    # 延续 Step 15：保留感知层
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
        mesh_prim_paths=["/World/defaultGroundPlane"],
        debug_vis=True,
    )

@configclass
class Go2ActionCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True 
    )

# 【剥洋葱核心 1】：新增指令配置
@configclass
class CommandsCfg:
    """定义随机速度指令发生器"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),  # 极其严谨的 API：每 3 秒重新采样一次新指令
        rel_standing_envs=0.0,             # 不允许站立，100% 的时间都要动
        rel_heading_envs=0.0,              # 关闭航向角控制，纯角速度控制
        debug_vis=True,                    # 可视化指令（UI中会在机器人头上画出红蓝箭头）
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),          # x 方向（前）：要求以 0.5 ~ 1.0 m/s 前进
            lin_vel_y=(0.0, 0.0),          # y 方向（横）：不横向走
            ang_vel_z=(-0.5, 0.5),         # z 轴旋转（偏航）：允许轻微左右转
        ),
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 关节感知 (12维)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        # 2. 地形感知 (121维)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5})
        
        # 【剥洋葱核心 2】：指令闭环 (3维: vx, vy, wz)
        # 将上面 CommandsCfg 生成的指令，作为环境观测喂给网络
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # 拼接洋葱: 12 + 121 + 3 = 136 维
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
    # 必须将指令配置挂载到总配置中
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4 
        self.episode_length_s = 20.0

def main():
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    
    print(f"--- 剥洋葱 Step 16 验证 ---")
    print(f"大脑已接收指令！总观测维度: {obs_dim} (12关节 + 121高度点 + 3速度指令)") 
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            obs_dict, _, _, _, _ = env.step(actions)
            
            # 实时验证 API：直接从 command_manager 中提取当前下达的指令
            current_command = env.command_manager.get_command("base_velocity")[0]
            
            print(f"当前目标指令 | 前进: {current_command[0]:.2f} m/s, 转向: {current_command[2]:.2f} rad/s", end='\r')
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()