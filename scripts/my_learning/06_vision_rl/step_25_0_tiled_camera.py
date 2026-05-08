"""
Step 25.0: TiledCamera 传感器 —— RGB 端到端 RL 的起点
================================================================
文件描述：在 CartPole 场景中加入 TiledCamera，读取 RGB 图像数据并打印形状、
          数值范围、数据类型，完全不涉及 RL 逻辑。

剥洋葱重点：
1. 三种相机类型的选择原则：
     Camera          → Omniverse GPU 渲染，单/少量相机场景适用
     TiledCamera     → 平铺并行渲染，多环境 RL 训练首选（本系列使用）
     RayCasterCamera → Warp 光线投射，仅深度/法线，不支持 RGB
2. TiledCamera 关键参数：
     prim_path  → 挂载到每个 ENV 下的 Camera prim
     offset     → 相对于 prim_path 父节点的位置和朝向（convention 指定坐标系）
     data_types → 按需选择（只选 "rgb" 可减少渲染开销）
     width/height → RL 训练常用 100×100（计算效率与信息量的平衡点）
3. AppLauncher 必须加 enable_cameras=True，否则相机渲染不会启用。
4. 图像数据：camera.data.output["rgb"] 形状 (N, H, W, 4)，dtype uint8，
   最后一维是 RGBA，其中 A 通道在 mdp.image 中会被自动丢弃。

注意事项：
  - 地面（ground plane）会挡住从 Z>0 俯视的相机，因此本步骤不加地面。
  - env_spacing=20 比默认值大，避免相邻环境的物体出现在相机视野中。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_25_0_tiled_camera.py
"""

# ==========================================
# 0. 启动引擎（enable_cameras=True 是关键）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


# ==========================================
# 1. 场景配置（CartPole + TiledCamera，无地面）
# ==========================================

@configclass
class CartpoleWithCameraSceneCfg(InteractiveSceneCfg):
    # 不加地面：地面会挡住从 Z>0 俯视的相机视线
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # 核心新知识点：TiledCamera 配置
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # 相机安装在距 CartPole -7m 处，高 3m，朝向 CartPole
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-7.0, 0.0, 3.0),
            rot=(0.9945, 0.0, 0.1045, 0.0),  # 约 12° 俯仰角，确保看到杆
            convention="world",              # world 坐标系（非 ROS/OpenGL）
        ),
        data_types=["rgb"],                  # 只请求 RGB，减少渲染开销
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=100,   # RL 训练常用 100×100
        height=100,
    )


# ==========================================
# 2. MDP 最小配置（仅用于启动环境，验证相机）
# ==========================================

@configclass
class MinimalActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )


@configclass
class MinimalObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class MinimalEventCfg:
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class MinimalRewardCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class MinimalTermCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CartpoleCameraEnvCfg(ManagerBasedRLEnvCfg):
    # env_spacing=20：相邻环境间距足够大，避免相机拍到邻居
    scene: CartpoleWithCameraSceneCfg = CartpoleWithCameraSceneCfg(num_envs=4, env_spacing=20.0)
    actions: MinimalActionsCfg = MinimalActionsCfg()
    observations: MinimalObsCfg = MinimalObsCfg()
    events: MinimalEventCfg = MinimalEventCfg()
    rewards: MinimalRewardCfg = MinimalRewardCfg()
    terminations: MinimalTermCfg = MinimalTermCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# ==========================================
# 3. 主函数：打印相机数据的形状和数值
# ==========================================

def main():
    env = ManagerBasedRLEnv(cfg=CartpoleCameraEnvCfg())

    print("\n" + "=" * 60)
    print("剥洋葱 Step 25.0 —— TiledCamera 传感器验证")
    print("=" * 60)

    # 获取相机对象
    camera = env.scene.sensors["tiled_camera"]

    # 运行几步让相机完成初始化
    for _ in range(5):
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)

    # 访问原始 RGB 数据（注意：这里是未经 mdp.image 归一化的原始数据）
    rgb_raw = camera.data.output["rgb"]

    print(f"\n[相机配置]")
    print(f"  分辨率: {camera.image_shape[1]} × {camera.image_shape[0]} (W × H)")
    print(f"  环境数: {env.num_envs}")

    print(f"\n[RGB 数据形状]")
    print(f"  camera.data.output['rgb'].shape = {rgb_raw.shape}")
    print(f"  解读: (num_envs={rgb_raw.shape[0]}, H={rgb_raw.shape[1]}, W={rgb_raw.shape[2]}, C={rgb_raw.shape[3]})")
    print(f"  格式: NHWC（N=批次, H=高, W=宽, C=通道数）")
    print(f"  注意: C=4 是 RGBA，第 4 个 A 通道（透明度）在 mdp.image 中会被自动丢弃")

    print(f"\n[数据类型与值域]")
    print(f"  dtype: {rgb_raw.dtype}")
    print(f"  值域: [{rgb_raw.min().item()}, {rgb_raw.max().item()}]")
    print(f"  说明: uint8 图像，0=黑，255=白")

    print(f"\n[三种相机对比]")
    print(f"  Camera          → Omniverse GPU 渲染，支持 RGB/分割，单相机场景")
    print(f"  TiledCamera     → 平铺并行渲染，多环境 RL 首选（本步骤使用）")
    print(f"  RayCasterCamera → Warp 光线投射，速度快，但仅支持深度/法线，不支持 RGB")

    print(f"\n✅ 相机数据读取成功！下一步（step_26_0）：将图像接入 ObservationManager。\n")

    # 实时显示（按 Ctrl+C 退出）
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)
            step += 1
            if step % 100 == 0:
                rgb = camera.data.output["rgb"]
                print(f"Step {step}: rgb min={rgb.min().item():.0f} max={rgb.max().item():.0f}", end="\r")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
