"""
Step 18.1: 训练宇树 Go2 平地行走控制器（完整修复版）
================================================================
文件描述：在 step_18_0 的最简闭环基础上，叠加 7 个工程优化点，达到生产质量的训练配置。
          建议先运行 step_18_0 理解基础流程，再对比本文件的差异。

剥洋葱重点（相比 step_18_0 新增的部分）：
1. 奖励从 2 条扩展到 9 条：加入角速度跟踪、垂直弹跳惩罚、电机保护、步态引导。
2. 观测从 30 维扩展到 48 维：加入 base_lin/ang_vel（6维）和 last_action（12维）。
3. RunningStandardScaler 归一化观测：消除各维度量纲差异，显著提升收敛稳定性。
4. ContactSensorCfg + illegal_contact：比高度判定更早检测跌倒，提升样本效率。

关键参数说明（与 naive 版本的 8 处差异）：
  scale=0.25    : 防止随机初始化时输出暴力动作导致出生即倒
  reset_root_state_uniform : 零速度贴地重置，避免高空坠落冲击
  illegal_contact          : 接触传感器检测 base 触地，样本效率高
  track_ang_vel_z_exp      : 转向指令必须有对应奖励，否则策略忽略转向
  lin_vel_z_l2             : 惩罚垂直弹跳，迫使策略学真正的步行
  feet_air_time            : 步态引导奖励，鼓励四腿交替迈步
  RunningStandardScaler    : 观测归一化，防止训练发散
  rel_standing_envs=0.1    : 10% 环境保持零速，先学站立再学行走

学习成果：
  训练 250 万步后（RTX 5070 约 12 小时），Go2 应能在平地稳定行走。
  日志保存至 logs/skrl/go2_opt1_training，可用 tensorboard 查看曲线。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_18_1_train_go2_full.py

后续：
  训练完成后运行 step_19_0_play_go2.py 查看推理效果。
"""

import math
import os

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ==========================================
# 0. 启动引擎（必须第一步，Isaac Sim 运行时尚未加载时不能 import 任何 isaaclab 子模块）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

# AppLauncher 启动后才能安全 import 依赖 pxr / Isaac Sim 的模块
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_loco

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


# ==========================================
# 1. 场景配置
# ==========================================

# 云端 S3 路径 404，使用本地 USD
LOCAL_UNITREE_GO2_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(
        usd_path="/home/hit-lgc/HomeWorkspace/unitree_rl_lab/unitree_model/Go2/usd/go2.usd"
    )
)


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    # [FIX 2/3] 使用平面地形（terrain_type="plane"），无需地形生成器
    # 配置足够的摩擦力，防止机器人在平地打滑
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = LOCAL_UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # [FIX 3/6] 全身接触传感器（prim_path 必须覆盖全部 19 个链接）：
    # - filter_prim_paths_expr 是过滤"与哪些外部物体接触"，不能用于限制传感器链接范围
    # - 需要减少 CPU 开销只能靠降低 history_length 或减少 update_period
    # - track_air_time=True 是 feet_air_time 奖励计算的必要条件
    # - history_length=3 让传感器能检测短暂接触
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


# ==========================================
# 2. 动作配置
# ==========================================

@configclass
class Go2ActionCfg:
    # [FIX 1] scale: 1.0 → 0.25
    # 官方 Go2 配置使用 0.25，大幅降低策略输出的实际关节位移幅度，
    # 避免网络随机初始化时就产生暴力动作导致翻倒
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ==========================================
# 3. 速度指令配置
# ==========================================

@configclass
class Go2CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # [FIX 8] rel_standing_envs=0.1：10% 的环境保持零速指令，
        # 让策略先学会站稳，再学行走，降低早期训练难度
        rel_standing_envs=0.1,
        # [FIX 8] 速度范围包含负值和零速，避免策略只会前冲而不能站立
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


# ==========================================
# 4. 观测配置
# ==========================================

@configclass
class Go2ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """48 维本体感知观测（3+3+3+3+12+12+12）"""
        # 基座速度（6维）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        # 投影重力向量（3维）—— IMU 平衡感，关键的姿态感知
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        # 速度指令（3维）
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # [FIX] joint_pos_rel：相对默认姿态的关节偏移量（12维），比原始绝对角度更有物理意义
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # [FIX] joint_vel_rel：相对默认速度的关节速度（12维）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # 上一帧动作（12维）—— 保证时序连贯性
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ==========================================
# 5. 奖励配置
# ==========================================

@configclass
class Go2RewardCfg:
    # --- 主任务奖励 ---

    # 线速度跟踪（权重提高到 1.5，参考官方 Go2 配置）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # [FIX 4] 新增：角速度跟踪奖励，引导机器人学习转向
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- 姿态稳定惩罚 ---

    # [FIX 5] 新增：惩罚身体垂直弹跳（跌倒时最明显的特征）
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # [FIX 5] 新增：惩罚身体俯仰和滚转角速度（防止前倾、侧翻）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 躯干水平惩罚（权重提高到 -2.5，参考官方 Go2 flat 配置）
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- 电机保护 / 平滑惩罚 ---

    # 力矩惩罚（权重参考官方 Go2：-2e-4）
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节加速度惩罚（抑制高频抖动）
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 动作变化率惩罚（防止策略输出突变，解决"乱扒"）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # [FIX 6] 步态引导：奖励脚部离地时间，鼓励四腿交替迈步而非"爬行"或"滑行"
    # body_names=".*_foot" 是 Go2 脚部链接的正则匹配
    feet_air_time = RewTerm(
        func=mdp_loco.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


# ==========================================
# 6. 终止条件配置
# ==========================================

@configclass
class Go2TermCfg:
    # [FIX 3] 使用接触传感器检测 base 触地：比高度阈值更早、更准确地发现跌倒
    # threshold=1.0N 意味着任何有效接触都会触发终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    # [FIX] time_out=True 告知值函数"这是时间耗尽而非跌倒"，
    # 让 PPO 正确进行价值函数自举（bootstrap），避免低估存活奖励
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ==========================================
# 7. 重置事件配置
# ==========================================

@configclass
class Go2EventCfg:
    # [FIX 2] 使用标准 reset_root_state_uniform：
    # - 在默认站立姿态附近小范围随机初始化位置和朝向
    # - velocity_range 全部为 (0, 0)：零速度重置，避免出生时有初速导致立即失稳
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # 关节在默认位置精确初始化（scale=1.0 表示无随机偏移），零初速度
    # 参考官方 Go2 配置：position_range=(1.0, 1.0)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ==========================================
# 8. 总环境配置
# ==========================================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    # RTX 5070 (12GB GDDR7)：2048 envs 时显存仅用 26%，扩到 4096 可提升 30-50% 吞吐量
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: Go2ObsCfg = Go2ObsCfg()
    commands: Go2CommandCfg = Go2CommandCfg()
    rewards: Go2RewardCfg = Go2RewardCfg()
    terminations: Go2TermCfg = Go2TermCfg()
    events: Go2EventCfg = Go2EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        # 物理仿真步长：0.005s = 200Hz，控制频率 = 200/4 = 50Hz
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # 接触传感器以物理步长频率更新（每步都采样）
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


# ==========================================
# 9. 主训练函数
# ==========================================

def main():
    print("\n[INFO] 1. 正在初始化 4096 只 Go2 并行物理环境...")
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print("[INFO] 2. 正在加载 PPO 训练配置...")
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
                    {"name": "net", "input": "OBSERVATIONS", "layers": [256, 128], "activations": "elu"}
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
            "rollouts": 24,
            "learning_epochs": 5,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-3,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            # [FIX 7] 观测归一化：
            # SKRL 2.0 的正确类名是 RunningStandardScaler（不是 RunningMeanStd）
            # Runner 在模块级已 import 该类，eval("RunningStandardScaler") 可正确解析
            # state_preprocessor 在 skrl 2.0 中已被 observation_preprocessor 取代，
            # Runner 会自动将其转换并注入 size/device，无需手动指定 size
            "state_preprocessor": "RunningStandardScaler",
            "state_preprocessor_kwargs": {},
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": {},
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 2.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            # [FIX] time_limit_bootstrap=True：episode 超时时用值函数估计剩余回报，
            # 而不是把超时当作终止死亡处理
            "time_limit_bootstrap": True,
            "experiment": {
                "directory": os.path.join(LOGS_DIR, "skrl/go2_opt1_training"),
                "experiment_name": "",
                "write_interval": "auto",
                "checkpoint_interval": "auto",
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            # RTX 5070 + 4096 envs 实测参考（基于 ~59 it/s 估算）：
            # - 今晚到明天11:30（12.5小时）：2_500_000   (~12.5 小时)
            # - 快速验证（看曲线是否上升）：8_000_000   (~38 小时)
            # - 正式训练（平地行走）      ：50_000_000  (~10 天)
            # - 高质量训练               ：100_000_000 (~20 天)
            "timesteps": 2_500_000,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    print("[INFO] 3. 启动 Runner，开始魔鬼训练（预计耗时 20-40 分钟）...")
    runner = Runner(env, experiment_cfg)
    runner.run()

    print(f"\n[SUCCESS] 训练完成！模型已保存至 {os.path.join(LOGS_DIR, 'skrl/go2_opt1_training')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
