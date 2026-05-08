"""
Step 17.1: 课程学习 (Curriculum) 与地形演进
================================================================
文件描述：在 step_17_0 的地形基础上新增 CurriculumManager，实现动态难度调整——
          表现好的机器人晋升到更难的行，表现差的降级到更容易的行。

剥洋葱重点：
1. 与 step_17_0 的唯一差异：添加 CurriculumCfg 和挂载到 Go2EnvCfg.curriculum。
2. CurriculumTermCfg 调用 terrain_levels_vel：根据实际速度 vs 指令速度的比值评分。
3. 架构解耦：地形生成（step_17_0）与难度调度（本步骤）是完全独立的概念。

学习成果：
  运行后终端实时打印各环境的地形难度等级，观察它如何随运动表现动态升降。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_17_1_curriculum.py
"""

# 0. 启动引擎
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils

# 全局 API 架构对齐：核心层 vs 任务层
import isaaclab.envs.mdp as mdp  
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_loco  

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg 
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm 
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# ==================== 1. 自定义判定逻辑 ====================

def my_term_fallen(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """自定义摔倒判定：当躯干 Z 轴高度低于 0.2 米时，判定为倒地。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < 0.2

# ==================== 2. 配置区 ====================

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    # 程序化地形矩阵
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10, 
            num_cols=20, 
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.1), step_width=0.3
                ),
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
            },
        ),
        max_init_terrain_level=5,
    )
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
        mesh_prim_paths=["/World/terrain"], 
        debug_vis=True,
    )

@configclass
class Go2ActionCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", 
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),   
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(-0.2, 0.2)   
        ),
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5})
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    track_lin_vel_x = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.25})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})

@configclass
class TerminationsCfg:
    base_contact = DoneTerm(func=my_term_fallen, params={"asset_cfg": SceneEntityCfg("robot")})

@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp_loco.terrain_levels_vel)

# ==================== 3. 环境总装 ====================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=16, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4 
        self.episode_length_s = 10.0

def main():
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    
    print(f"--- 剥洋葱 Step 17 验证 ---")
    print(f"地形矩阵与课程学习已打通！所有参数校验通过。")
    print(f"地形总难度阶梯 (Rows): {env.scene.terrain.cfg.terrain_generator.num_rows}")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)
            
            # 终极修复：直接从 terrain 对象获取难度等级数据
            current_levels = env.scene.terrain.terrain_levels
            print(f"环境 0-4 的地形难度 Level: {current_levels[:5].tolist()}", end='\r')
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()