"""
Step 22.0: USD 相机——RGB 图像观测 (USD Camera)
================================================================
文件描述：配置基于物理渲染的 USD 相机，采集 RGB / 深度 / 语义分割图像，
          演示完整的视觉观测管道（渲染 → 张量 → 网络输入）。

剥洋葱重点：
1. CameraCfg（USD 相机）支持光线追踪渲染，图像质量高于 RayCast 相机。
2. camera.data.output["rgb"] 形状为 (num_envs, H, W, 4)（RGBA）。
3. 多相机并行渲染是 GPU 显存大户，大规模训练时需权衡分辨率与 num_envs。

学习成果：
  运行后能将 RGB 图像张量保存为 PNG，直观验证视觉管道完整性。

参考官方教程：scripts/tutorials/04_sensors/run_usd_camera.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/05_advanced/step_22_camera_usd.py

待实现：参照 scripts/tutorials/04_sensors/run_usd_camera.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/04_sensors/run_usd_camera.py 实现
pass
