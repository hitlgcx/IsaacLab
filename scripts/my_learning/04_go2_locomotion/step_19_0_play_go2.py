"""
Step 19.0: Go2 运动控制器推理展示（对应 step_18_1_train_go2_full.py）
================================================================
文件描述：加载 Step 18 训练好的 Go2 策略，用图形界面展示平地行走效果。

剥洋葱重点：
1. 推理配置必须与训练 100% 对齐（观测顺序、函数名、scale、dt 均不可变）。
2. enable_training_mode(False) 关闭探索噪声，使用 mean_actions 展示最稳定步态。
3. env.unwrapped.cfg.sim.dt × decimation 计算真实控制步长，用于实时速度同步。

推理与训练的关键区别：
  num_envs=4（便于观察）vs 训练时 4096
  headless=False（开启 UI）vs 训练时 True
  timesteps=0（不训练，直接推理）
  write_interval=0, checkpoint_interval=0（不写日志）

学习成果：
  运行后 4 只 Go2 展示已训练步态，在 UI 中对焦 env_0/Robot/base 观察行走。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_19_0_play_go2.py  （需先完成 step_18 训练）

前置依赖：
  step_18_train_go2_opt1.py（生成 logs/skrl/go2_opt1_training 模型文件）
"""

# ==========================================
# 0. 启动引擎（必须第一步，AppLauncher 必须在 torch/cuda 之前）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False}) 
simulation_app = app_launcher.app

import os
import glob
import sys
import time
import math
import torch

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

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
# 1. 场景配置（严格对齐训练）
# ==========================================

# 云端 S3 路径 404，使用本地 USD
LOCAL_UNITREE_GO2_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(
        usd_path="/home/hit-lgc/HomeWorkspace/unitree_rl_lab/unitree_model/Go2/usd/go2.usd"
    )
)


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
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

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


# ==========================================
# 2. 动作配置（严格对齐训练：scale=0.25）
# ==========================================

@configclass
class Go2ActionCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ==========================================
# 3. 速度指令配置（推理时用固定前进指令，便于观察步态）
# ==========================================

@configclass
class Go2CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


# ==========================================
# 4. 观测配置（必须与训练 100% 对齐：顺序、函数名均不可变）
# ==========================================

@configclass
class Go2ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 顺序和函数名与 step_18_train_go2_opt1.py 完全一致（共 48 维）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ==========================================
# 5. 奖励配置（推理时不参与计算，但必须保留以通过配置校验）
# ==========================================

@configclass
class Go2RewardCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
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
# 6. 终止条件配置（严格对齐训练：接触传感器检测 base 触地）
# ==========================================

@configclass
class Go2TermCfg:
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ==========================================
# 7. 重置事件配置（严格对齐训练）
# ==========================================

@configclass
class Go2EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ==========================================
# 8. 总环境配置（推理用 4 envs，其余与训练一致）
# ==========================================

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4, env_spacing=4.0)
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: Go2ObsCfg = Go2ObsCfg()
    commands: Go2CommandCfg = Go2CommandCfg()
    rewards: Go2RewardCfg = Go2RewardCfg()
    terminations: Go2TermCfg = Go2TermCfg()
    events: Go2EventCfg = Go2EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


# ==========================================
# 9. 工具函数
# ==========================================

def get_latest_model_path(search_dir: str = "") -> str:
    if not search_dir:
        search_dir = os.path.join(LOGS_DIR, "skrl/go2_opt1_training")
    pt_files = glob.glob(f"{search_dir}/**/*.pt", recursive=True)
    if not pt_files:
        print(f"\n[ERROR] 未在 {search_dir} 找到 .pt 模型文件。")
        print("[ERROR] 请确认 step_18_train_go2_opt1.py 已完成训练。")
        sys.exit(1)
    latest = max(pt_files, key=os.path.getctime)
    return latest


# ==========================================
# 10. 主推理函数
# ==========================================

def main():
    print("\n[INFO] 1. 正在初始化 4 只 Go2 推理环境...")
    env = ManagerBasedRLEnv(cfg=Go2EnvCfg())
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # 网络结构必须与训练完全一致
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
            "state_preprocessor": "RunningStandardScaler",
            "state_preprocessor_kwargs": {},
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": {},
            "experiment": {
                "directory": os.path.join(LOGS_DIR, "skrl/go2_opt1_play"),
                "experiment_name": "",
                "write_interval": 0,
                "checkpoint_interval": 0,
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 0,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }

    print("[INFO] 2. 正在搜索最新模型权重...")
    runner = Runner(env, experiment_cfg)
    model_path = get_latest_model_path()
    print(f"[INFO] 加载模型：{model_path}")
    runner.agent.load(model_path)

    # 关闭训练模式：使用均值动作（无噪声），展示最稳定步态
    runner.agent.enable_training_mode(False, apply_to_models=True)

    dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    obs, _ = env.reset()
    states = env.state()

    print("\n[INFO] 3. 开始播放！请在 Isaac Sim UI 中选中 env_0/Robot/base 并按 'F' 对焦。")
    print("[INFO] 指令：前进 0.5~1.0 m/s，左右转 ±0.5 rad/s，10秒换一次指令。\n")

    step = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            states = env.state()

        if terminated.any():
            fall_count = terminated.sum().item()
            print(f"[step {step:6d}] {fall_count} 只狗跌倒重置（训练前期正常）")

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
