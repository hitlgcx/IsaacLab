"""
Step 1.1: 通过 ./isaaclab.sh 运行的场景示例（argparse 模式）
================================================================
文件描述：演示官方推荐的命令行参数传递方式，与 step_1_scene.py 内容相同但运行模式不同。

与 step_1_scene.py 的核心区别：

  step_1_scene.py（直接 python 运行）：
    app_launcher = AppLauncher({"headless": False})  # 参数硬编码在脚本里
    运行：python scripts/my_learning/01_sim_basics/step_1_scene.py

  step_1_1_scene_cli.py（./isaaclab.sh 运行）：
    app_launcher = AppLauncher(args_cli)             # 参数来自命令行
    运行：./isaaclab.sh -p scripts/my_learning/01_sim_basics/step_1_1_scene_cli.py --headless

./isaaclab.sh 模式的优势：
  - 支持从命令行动态传入 --headless、--device、--enable_cameras 等参数
  - 与 Isaac Sim 官方教程脚本风格一致
  - 便于 CI/CD 和批量测试（无头模式自动化运行）

剥洋葱重点：
1. argparse + AppLauncher.add_app_launcher_args() 是标准参数注册方式。
2. AppLauncher(args_cli) 接收解析后的 Namespace 对象，而非字典。
3. Isaac Sim 相关模块必须在 AppLauncher 启动后才能导入（与 step_1 相同）。

学习成果：
  运行后你将看到 4 台倒立摆小车在重力作用下掉落在地面上（与 step_1 效果相同）。
  尝试加上 --headless 参数观察无头模式的效果。

运行方式：
  # 图形界面（默认）
  ./isaaclab.sh -p scripts/my_learning/01_sim_basics/step_1_1_scene_cli.py

  # 无头模式（不开窗口，适合训练/CI）
  ./isaaclab.sh -p scripts/my_learning/01_sim_basics/step_1_1_scene_cli.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# ==========================================
# 1. 命令行参数解析（./isaaclab.sh 模式的标准写法）
# ==========================================
parser = argparse.ArgumentParser(description="Step 1.1: Isaac Lab 场景创建示例（CLI 参数模式）")

# 将 Isaac Sim / Isaac Lab 的标准命令行参数注册到 parser
# 包括：--headless, --device, --enable_cameras 等
AppLauncher.add_app_launcher_args(parser)

# 解析参数（./isaaclab.sh 会自动将剩余参数传给脚本）
args_cli = parser.parse_args()

# 用解析后的参数启动 AppLauncher（与 step_1 的字典方式等价，但更灵活）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# 2. AppLauncher 启动后才能安全导入 isaaclab 子模块
# ==========================================
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# ==========================================
# 3. 场景配置（与 step_1_scene.py 完全相同）
# ==========================================

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """与 step_1 相同的场景图纸，此处不重复注释"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0),
    )

# ==========================================
# 4. 运行逻辑（与 step_1_scene.py 完全相同）
# ==========================================

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] 场景加载完成！")
    print(f"[INFO] 当前运行模式: {'无头模式' if args_cli.headless else '图形界面'}")

    while simulation_app.is_running():
        sim.step()
        scene.update(dt=sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
