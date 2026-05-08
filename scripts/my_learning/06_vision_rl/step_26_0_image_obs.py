"""
Step 26.0: 图像观测 —— mdp.image 集成进 ObservationManager
================================================================
文件描述：在 step_25_0 的 TiledCamera 基础上，将图像接入 ObservationManager，
          让 RL 策略的"感官"从状态向量切换为像素图像。

剥洋葱重点：
1. 与 step_25_0 的唯一差异：PolicyCfg 中用 mdp.image 替换 mdp.joint_pos_rel。
2. mdp.image 的 normalize=True 行为：
     RGB → 除以 255 转 float32 → 减去当前批次均值（白化）
     白化的目的：消除场景整体亮度对观测的影响，让网络更关注结构信息。
3. 图像观测的 shape：(num_envs, H, W, 3)，格式 NHWC，注意 C=3（RGBA 自动剔除 A）。
4. 为何图像不能与状态向量直接 concatenate？
     状态向量 shape: (num_envs, D)  → 2D
     图像观测 shape: (num_envs, H, W, C) → 4D
     两者维度不同，无法拼接。必须"只用图像"或"分开处理"。
     （concatenate_terms=True 在纯图像 ObsGroup 中是允许的，因为只有一个 term）

学习成果：
  运行后打印 obs["policy"] 的完整信息，理解图像进入 RL 环境后的格式。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_26_0_image_obs.py
"""

import math

# ==========================================
# 0. 启动引擎
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import torch
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
# 1. 场景（与 step_25_0 完全相同）
# ==========================================

@configclass
class CartpoleWithCameraSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100, height=100,
    )


# ==========================================
# 2. 动作 / 奖励 / 终止（最小配置，用于启动 RL 环境）
# ==========================================

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )

@configclass
class EventCfg:
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)},
    )

@configclass
class RewardCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

# ==========================================
# 3. 核心新知识点：图像观测 ObsGroup
# ==========================================

@configclass
class ImageObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """
        与 step_9_0 的 PolicyCfg 对比（step_9_0 是状态观测）：

        状态观测（step_9_0）：
            joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)   # shape: (N, 2)
            joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)   # shape: (N, 2)
            合并后 shape: (N, 4)

        图像观测（本步骤）：
            image = ObsTerm(func=mdp.image, ...)              # shape: (N, 100, 100, 3)
            不存在合并问题（单个 term，4D tensor）

        normalize=True 效果：
            原始 uint8 (0-255) → float32 / 255 → 减去批次均值
            白化后值域约 (-0.5, 0.5)，均值为 0
        """
        image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "normalize": True,   # 除以 255 + 减均值（白化）
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # 单个 term，concatenate 不影响 shape

    policy: PolicyCfg = PolicyCfg()


# ==========================================
# 4. 环境总装
# ==========================================

@configclass
class CartpoleCameraEnvCfg(ManagerBasedRLEnvCfg):
    scene: CartpoleWithCameraSceneCfg = CartpoleWithCameraSceneCfg(num_envs=4, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ImageObsCfg = ImageObsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationCfg = TerminationCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# ==========================================
# 5. 主函数：验证图像观测的 shape 和值域
# ==========================================

def main():
    env = ManagerBasedRLEnv(cfg=CartpoleCameraEnvCfg())

    print("\n" + "=" * 60)
    print("剥洋葱 Step 26.0 —— 图像观测集成验证")
    print("=" * 60)

    # reset 触发第一次观测
    obs, _ = env.reset()

    image_obs = obs["policy"]

    print(f"\n[观测 shape 对比]")
    print(f"  step_9_0 状态观测: (num_envs, 4)     → 2D tensor")
    print(f"  step_26_0 图像观测: {tuple(image_obs.shape)} → 4D tensor (NHWC)")
    print(f"  N={image_obs.shape[0]}, H={image_obs.shape[1]}, W={image_obs.shape[2]}, C={image_obs.shape[3]}")
    print(f"  C=3（RGBA 原始数据，A 通道在 mdp.image 内部自动丢弃）")

    print(f"\n[normalize=True 效果]")
    print(f"  dtype: {image_obs.dtype}  （float32，原始 uint8 已除以 255）")
    print(f"  值域: [{image_obs.min().item():.4f}, {image_obs.max().item():.4f}]")
    print(f"  均值: {image_obs.mean().item():.6f}  （接近 0，白化生效）")

    print(f"\n[为何不能与状态向量 concatenate？]")
    print(f"  状态向量 shape: (N, D) → 2D，可以 cat")
    print(f"  图像观测 shape: (N, H, W, C) → 4D，维度不匹配，无法 cat")
    print(f"  解决方案：要么只用图像，要么分两个 ObsGroup 分别处理")

    print(f"\n✅ 图像观测验证成功！下一步（step_27_0）：配置 CNN 策略网络。\n")

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            obs, _, _, _, _ = env.step(actions)
            step += 1
            if step % 50 == 0:
                img = obs["policy"]
                print(f"Step {step}: obs shape={tuple(img.shape)}, mean={img.mean():.4f}", end="\r")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
