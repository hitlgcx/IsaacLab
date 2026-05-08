"""
Step 27.0: CNN 策略配置 —— NHWC→NCHW 与卷积层设计
================================================================
文件描述：在 step_26_0 的图像观测基础上，展示如何为 skrl 配置 CNN 策略网络，
          重点理解 NHWC→NCHW 格式转换和卷积层输出维度的计算。
          运行 100 步验证网络能正确 forward，不追求任何收敛效果。

剥洋葱重点：
1. Isaac Lab 返回的图像格式是 NHWC（PyTorch 习惯的 NCHW 的转置）。
   PyTorch Conv2d 期望 NCHW，因此必须先 permute：
     NHWC (N, 100, 100, 3) → NCHW (N, 3, 100, 100)
   skrl 的 network 配置中用 "permute(OBSERVATIONS, (0, 3, 1, 2))" 实现这一转换。

2. 三层卷积的输出维度计算（100×100 输入）：
     输出尺寸 = floor((输入 - kernel) / stride) + 1
     Conv1: (100-8)/4 + 1 = 24  → (32, 24, 24)
     Conv2: (24-4)/2  + 1 = 11  → (64, 11, 11)
     Conv3: (11-3)/1  + 1 = 9   → (64, 9, 9)
   Flatten 后: 64 × 9 × 9 = 5184 维

3. CNN 后接 MLP：features_extractor → net → ACTIONS
   policy 和 value 共用同一套 CNN 配置（separate=False）。

4. 对比 step_9_0 的 MLP 策略（输入 4 维状态 → [64, 64] MLP）：
   CNN 策略参数量更大，但能直接处理像素输入。

学习成果：
  运行后打印网络成功构建的确认信息，以及每层的输出维度推算。
  不需要等待收敛，只需确认 "forward pass 不报错" 即达到本步骤目标。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/06_vision_rl/step_27_0_cnn_policy.py
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
# 1. 场景 + MDP 配置（与 step_26_0 相同）
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

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )

@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": True},
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
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

@configclass
class TermCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_oob = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleCameraEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=64, env_spacing=20.0)
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
# 2. 核心新知识点：CNN 策略配置
# ==========================================

def make_cnn_experiment_cfg(logs_dir: str) -> dict:
    """
    CNN 策略配置详解：

    输入格式转换：
        NHWC (N, 100, 100, 3) --permute(0,3,1,2)--> NCHW (N, 3, 100, 100)
        为什么需要 permute？PyTorch 的 Conv2d 期望 NCHW 格式，
        而 Isaac Lab 的 mdp.image 返回 NHWC 格式（Height-Width-Channel）。

    卷积层输出维度（100×100 输入）：
        output_size = floor((input - kernel) / stride) + 1
        Conv1: (100-8)/4 + 1 = 24  → feature map: (32, 24, 24)
        Conv2: (24-4)/2  + 1 = 11  → feature map: (64, 11, 11)
        Conv3: (11-3)/1  + 1 = 9   → feature map: (64, 9,  9)
        Flatten: 64 × 9 × 9 = 5184 维

    MLP head：
        5184 → 512 → action_dim (1)
    """
    return {
        "seed": 42,
        "models": {
            "separate": False,  # policy 和 value 共享 CNN 参数
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "network": [
                    {
                        # 第 1 步：NHWC → NCHW（关键！PyTorch Conv2d 需要 NCHW）
                        "name": "features_extractor",
                        "input": "permute(OBSERVATIONS, (0, 3, 1, 2))",
                        "layers": [
                            # Conv1: 3→32 通道，kernel=8，stride=4
                            # 输出: (N, 32, 24, 24)
                            {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                            # Conv2: 32→64 通道，kernel=4，stride=2
                            # 输出: (N, 64, 11, 11)
                            {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                            # Conv3: 64→64 通道，kernel=3，stride=1
                            # 输出: (N, 64, 9, 9)
                            {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                            # Flatten: (N, 64, 9, 9) → (N, 5184)
                            "flatten",
                        ],
                        "activations": "relu",
                    },
                    {
                        # 第 2 步：MLP head，5184 → 512 → action
                        "name": "net",
                        "input": "features_extractor",
                        "layers": [512],
                        "activations": "elu",
                    },
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {
                        "name": "features_extractor",
                        "input": "permute(OBSERVATIONS, (0, 3, 1, 2))",
                        "layers": [
                            {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                            {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                            "flatten",
                        ],
                        "activations": "relu",
                    },
                    {"name": "net", "input": "features_extractor", "layers": [512], "activations": "elu"},
                ],
                "output": "ONE",
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 16,
            "learning_epochs": 4,
            "mini_batches": 2,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-4,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "state_preprocessor": None,
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
                "directory": os.path.join(logs_dir, "skrl/cartpole_rgb_cnn_verify"),
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            # 只运行 100 步，目的是验证 CNN 能正确 forward，不追求收敛
            "timesteps": 100,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }


# ==========================================
# 3. 主函数：打印 CNN 维度分析 + 验证 forward
# ==========================================

def main():
    print("\n" + "=" * 60)
    print("剥洋葱 Step 27.0 —— CNN 策略配置与维度分析")
    print("=" * 60)

    print("\n[卷积层输出维度计算]（输入: 100×100，3 通道）")
    print("  公式: output = floor((input - kernel) / stride) + 1")
    print("  Conv1: (100-8)/4 + 1 = 24  → feature map (32, 24, 24)")
    print("  Conv2: (24-4)/2  + 1 = 11  → feature map (64, 11, 11)")
    print("  Conv3: (11-3)/1  + 1 = 9   → feature map (64,  9,  9)")
    print("  Flatten: 64 × 9 × 9 = 5184 维")
    print("  MLP: 5184 → 512 → 1（action）")

    print("\n[初始化环境]...")
    env = ManagerBasedRLEnv(cfg=CartpoleCameraEnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print("[构建 CNN 策略并运行 100 步验证...]")
    experiment_cfg = make_cnn_experiment_cfg(LOGS_DIR)
    runner = Runner(env, experiment_cfg)
    runner.run()

    print("\n✅ CNN forward pass 验证成功！")
    print("   网络结构: NHWC → permute → NCHW → Conv×3 → Flatten(5184) → MLP(512) → action")
    print("   下一步（step_28_0）：用完整训练步数（50,000）观察视觉 RL 的收敛过程。")


if __name__ == "__main__":
    main()
    simulation_app.close()
