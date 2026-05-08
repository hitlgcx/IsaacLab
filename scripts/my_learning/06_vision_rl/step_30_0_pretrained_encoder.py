"""
Step 30.0: 预训练编码器 —— ResNet18 冻结特征替换 CNN
================================================================
文件描述：用 mdp.image_features（ResNet18 冻结特征）替换 mdp.image（原始像素），
          展示迁移学习在视觉 RL 中的应用，对比两种方案的差异。

剥洋葱重点：
1. 与 step_28_0 的唯一差异：ObsCfg 中用 mdp.image_features 替换 mdp.image。
   - mdp.image        → 返回 (N, 100, 100, 3)，需要 CNN 提取特征
   - mdp.image_features → 返回 (N, 512)，ResNet18 已提取好的 512 维特征向量

2. 观测从 4D NHWC 变为 2D 向量后，策略网络从 CNN+MLP 退化为普通 MLP：
   - step_28_0 策略: CNN(3→32→64→64, 5184 flatten) → MLP(512) → action
   - step_30_0 策略: MLP(512→256) → action（无 CNN 层，无 permute）

3. 两种方案的权衡：
   ┌─────────────────┬──────────────────────────┬───────────────────────────┐
   │                 │ CNN from scratch (28_0)  │ Pretrained ResNet (30_0)  │
   ├─────────────────┼──────────────────────────┼───────────────────────────┤
   │ 样本效率        │ 低（需从像素学特征）      │ 高（复用 ImageNet 特征）   │
   │ 收敛速度        │ 慢                        │ 快                        │
   │ 特征可优化性    │ 端到端可优化              │ 冻结，不随任务更新         │
   │ 显存占用        │ 低（CNN 轻量）            │ 较高（ResNet18 常驻显存）  │
   │ 适用场景        │ 任务特定视觉特征          │ 通用视觉特征（物体识别等） │
   └─────────────────┴──────────────────────────┴───────────────────────────┘

4. mdp.image_features 内部流程：
   raw_rgb (uint8) → normalize → resize to (224, 224) → ResNet18 → 512 维特征
   ResNet18 权重从 torchvision 加载，训练时自动冻结（requires_grad=False）。

注意：
  - 需要联网下载 ResNet18 权重（首次运行）。
  - 若无网络，可指定 model_device="cpu" 并预先下载权重到本地。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_30_0_pretrained_encoder.py
"""

import os
import math

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ==========================================
# 0. 启动引擎
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

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
from isaaclab_tasks.manager_based.classic.cartpole.mdp import joint_pos_target_l2
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


# ==========================================
# 1. 场景配置（与 step_28_0 相同）
# ==========================================

@configclass
class SceneCfg(InteractiveSceneCfg):
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
# 2. MDP 配置
# ==========================================

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )

@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """
        核心新知识点：用 mdp.image_features 替换 mdp.image

        mdp.image（step_28_0）：
          → 返回原始像素 shape: (N, 100, 100, 3)，策略网络需要 CNN 处理

        mdp.image_features（本步骤）：
          → 返回 ResNet18 特征向量 shape: (N, 512)，策略网络只需 MLP
          → 内部自动：raw_uint8 → /255 → resize(224×224) → ResNet18 → pool → 512 维
          → model_name="resnet18"：使用 torchvision 的 ResNet18（ImageNet 预训练）
          → 权重自动冻结（requires_grad=False）
        """
        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",    # 可换为 "resnet50"、"theia-tiny-patch16-224-cddsv" 等
                "model_device": "cuda:0",    # ResNet18 在 GPU 上运行（与仿真同设备）
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   # 512 维向量，2D tensor，可正常 concatenate

    policy: PolicyCfg = PolicyCfg()

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
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=joint_pos_target_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )

@configclass
class TermCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleResNetEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=20.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TermCfg = TermCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# ==========================================
# 3. 主训练函数
# ==========================================

def main():
    print("\n[INFO] 初始化环境（ResNet18 预训练编码器）...")
    env = ManagerBasedRLEnv(cfg=CartpoleResNetEnvCfg())
    obs_shape = env.observation_manager.group_obs_dim["policy"]
    print(f"[INFO] 观测 shape: {obs_shape}")
    print(f"[INFO] 对比：step_28_0 图像观测 shape: (100, 100, 3)")
    print(f"[INFO] ResNet18 已将 100×100 图像压缩为 512 维特征向量")
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # 策略网络：普通 MLP（无需 CNN，无需 permute）
    experiment_cfg = {
        "seed": 42,
        "models": {
            "separate": False,
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "network": [
                    {
                        # 观测已经是 512 维向量，直接送入 MLP，无需 permute 和 CNN
                        "name": "net",
                        "input": "OBSERVATIONS",   # 对比 step_28_0：这里不需要 permute
                        "layers": [256, 128],      # 比 step_28_0 的 MLP 更小（特征已经很好了）
                        "activations": "elu",
                    }
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {"name": "net", "input": "OBSERVATIONS", "layers": [256, 128], "activations": "elu"}
                ],
                "output": "ONE",
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 64,
            "learning_epochs": 4,
            "mini_batches": 8,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-4,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "state_preprocessor": "RunningStandardScaler",  # 对 512 维特征归一化
            "state_preprocessor_kwargs": None,
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": None,
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.0,
            "value_loss_scale": 1.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
            "experiment": {
                "directory": os.path.join(LOGS_DIR, "skrl/cartpole_resnet"),
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 100_000,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    print("[INFO] 启动训练（ResNet18 特征 + MLP 策略，100,000 步）...")
    print(f"[INFO] 日志目录: {os.path.join(LOGS_DIR, 'skrl/cartpole_resnet')}")
    print("[INFO] 预期：比 step_28_0（CNN from scratch）收敛更快，但编码器不随任务优化")
    runner = Runner(env, experiment_cfg)
    runner.run()

    print("\n[SUCCESS] 完成！")
    print("  对比两个 tensorboard 日志观察收敛速度差异：")
    print(f"  tensorboard --logdir {os.path.join(LOGS_DIR, 'skrl')}")
    print("  cartpole_rgb    ← step_28_0（CNN from scratch）")
    print("  cartpole_resnet ← step_30_0（预训练 ResNet18）")


if __name__ == "__main__":
    main()
    simulation_app.close()
