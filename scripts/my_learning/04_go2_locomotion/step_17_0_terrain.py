"""
Step 17.0: 程序化地形生成 (Procedural Terrain)
================================================================
文件描述：将 Go2 的训练地面从平面升级为程序化生成的地形矩阵，
          理解 TerrainImporterCfg 的分级结构，为 step_17_1 的课程学习做铺垫。

剥洋葱重点：
1. TerrainImporterCfg(terrain_type="generator") 生成 num_rows × num_cols 的地形网格。
2. sub_terrains 字典组合多种地形类型（阶梯 + 平地），proportion 控制占比。
3. 本步骤只生成地形，不含课程管理器——机器人被随机放置在地形上，无难度感知。
   （对比 step_17_1：课程管理器会根据表现动态选择放置的行）

学习成果：
  运行后可在 UI 中看到 10×20 的分级地形矩阵（阶梯与平面交错），
  高度扫描仪在地形上显示红色点阵，随机分布于所有难度区域。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_17_0_terrain.py
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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# ==================== 1. 自定义判定逻辑 ====================

def my_term_fallen(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """摔倒判定：躯干 Z 轴高度低于 0.2 米时判定为倒地。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < 0.2

# ==================== 2. 配置区 ====================

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    # 程序化地形矩阵（10 个难度行 × 20 个变体列）
    # 注意：此处仅生成地形，机器人被均匀随机分配到所有行
    # step_17_1 的 CurriculumManager 会根据成绩动态选择行
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,   # 10 个难度等级
            num_cols=20,   # 每级 20 个变体
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # 50% 台阶地形（难度高）
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.1), step_width=0.3
                ),
                # 50% 平面地形（难度低）
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
            },
        ),
        max_init_terrain_level=5,  # 初始最高在第 5 行（中等难度）
    )
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # 高度扫描传感器（mesh_prim_paths 需改为地形路径）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
        mesh_prim_paths=["/World/terrain"],
        debug_vis=True,
    )


@configclass
class Go2ActionCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.2, 0.2),
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )
        velocity_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    base_contact = DoneTerm(
        func=my_term_fallen, params={"asset_cfg": SceneEntityCfg("robot")}
    )


# ==================== 3. 环境总装（无 CurriculumCfg）====================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=16, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # 注意：这里没有 curriculum 字段——对比 step_17_1

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0


def main():
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())

    cfg = env.scene.terrain.cfg.terrain_generator
    print(f"--- 剥洋葱 Step 17.0 验证 ---")
    print(f"地形矩阵尺寸: {cfg.num_rows} 行 × {cfg.num_cols} 列")
    print(f"子地形类型: {list(cfg.sub_terrains.keys())}")
    print(f"机器人初始最高行: {env.scene.terrain.cfg.max_init_terrain_level}")
    print(f"观测维度: {env.observation_manager.group_obs_dim['policy'][0]}")
    print(f"（无课程管理器：机器人随机分布在所有难度区，对比 step_17_1 的动态分配）")

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(
                env.num_envs, env.action_manager.total_action_dim, device=env.device
            )
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
