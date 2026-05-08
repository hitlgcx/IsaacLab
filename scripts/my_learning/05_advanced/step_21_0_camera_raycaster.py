"""
Step 21.0: RayCast 相机——视觉 RL 基础 (RayCaster Camera)
================================================================
文件描述：配置 RayCasterCameraCfg 生成深度图/法线图，演示如何将图像观测
          接入 RL 管道，理解视觉 RL 与状态 RL 的关键区别。

剥洋葱重点：
1. RayCasterCamera 基于射线投射，比 USD 相机速度快，适合大规模并行训练。
2. 输出 data.output["distance_to_image_plane"]（深度图）形状为 (num_envs, H, W)。
3. 与 RayCaster（step_15）的区别：RayCasterCamera 输出 2D 图像而非 1D 高度数组。

学习成果：
  运行后能在终端看到深度图的统计信息（均值/方差），理解视觉传感器的数据格式。

参考官方教程：scripts/tutorials/04_sensors/run_ray_caster_camera.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/05_advanced/step_21_camera_raycaster.py

待实现：参照 scripts/tutorials/04_sensors/run_ray_caster_camera.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/04_sensors/run_ray_caster_camera.py 实现
pass
