"""
Step 12.0: 导入宇树 Go2 机器狗
================================================================
文件描述：从 CartPole 切换到真实四足机器人——用官方图纸加载 Go2，观察其关节结构。

剥洋葱重点：
1. 机器人切换只需替换 CFG：用 UNITREE_GO2_CFG 替换 CARTPOLE_CFG，其余框架代码不变。
2. Go2 有 12 个驱动关节（前/后腿 × 髋/大腿/小腿 × 左/右），命名规则在终端打印可见。
3. sim.reset() 之后才能读取物理视图数据（joint_names 等）。

学习成果：
  运行后你将看到一只 Go2 矗立在场景中，终端打印其 12 个驱动关节名称。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_12_import_go2.py
"""

# ==========================================
# 0. 启动引擎
# ==========================================
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# ==========================================
# 1. 配置区 (图纸设计)
# ==========================================
@configclass
class DogSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # Go2 是黑色的，适当提高灯光强度
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0)
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# ==========================================
# 2. 运行区 (施工预览)
# ==========================================
def main():
    # 初始化仿真上下文 (设定步长)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 实例化配置
    scene_cfg = DogSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # 全局重置：启动物理世界，建立物理视图（此后才能读取 joint_names 等）
    sim.reset()
    
    print("[INFO] 施工完成！宇树 Go2 已经准备就绪。")
    
    # 检查我们心心念念的关节名称
    # 注意：此时因为 sim.reset() 已经执行，物理视图已建立，可以直接读取
    print(f"机器狗 12 个驱动关节: {scene['robot'].joint_names}")

    # 保持窗口开启
    while simulation_app.is_running():
        # 物理步进
        sim.step()
        # 更新场景
        scene.update(dt=sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()