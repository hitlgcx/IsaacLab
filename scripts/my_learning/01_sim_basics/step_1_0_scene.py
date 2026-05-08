"""
Step 1.0: 搭建第一个仿真场景 (CartPole)
================================================================
文件描述：用最少的代码在 Isaac Lab 中建立一个包含 4 台倒立摆小车的仿真世界，
          理解场景搭建的骨架：AppLauncher → SimulationContext → InteractiveScene → step()。

剥洋葱重点：
1. AppLauncher({"headless": False}) 是硬编码字典模式，参数写死在脚本中（对比 step_1_1 的 argparse 模式）。
2. @configclass 是 Isaac Lab 的配置系统，用 Python 类描述场景"图纸"，运行时才实例化。
3. InteractiveScene 负责实际加载 USD 资产；sim.step() 推进一帧物理计算。

学习成果：
  运行后看到 4 台倒立摆小车在重力作用下掉落在地面上，窗口正常弹出即验证成功。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/01_sim_basics/step_1_0_scene.py
"""

# ==========================================
# 0. 启动引擎（必须第一步，任何 isaaclab 子模块 import 之前）
# ==========================================
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

# ==========================================
# 1. AppLauncher 启动后才能安全导入 isaaclab 子模块
# ==========================================
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==========================================
# 2. 场景配置（图纸）
# ==========================================

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """4 台倒立摆小车的场景图纸：地面 + 机器人 × 4 + 灯光"""

    # 地面平面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # 倒立摆小车：{ENV_REGEX_NS} 是 Isaac Lab 自动展开为多环境路径的占位符
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 全局照明
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0),
    )

# ==========================================
# 3. 主函数
# ==========================================

def main():
    # 物理仿真参数：dt=0.01s → 100Hz
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 按图纸实例化场景（num_envs=4 生成 4 个独立的环境副本）
    scene_cfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] 场景加载完成！4 台小车已就位。")

    while simulation_app.is_running():
        # 推进一帧物理计算（200Hz 物理更新）
        sim.step()
        # 通知场景内所有资产同步最新物理状态
        scene.update(dt=sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
