"""
Step 15.2: 帧变换传感器——足部世界坐标追踪 (Frame Transformer)
================================================================
文件描述：使用 FrameTransformerCfg 将 Go2 足部末端链接（FL_foot 等）的位姿
          实时转换到世界坐标系，为步态奖励（足部高度、足部速度）提供精确数据。

剥洋葱重点：
1. FrameTransformerCfg 以 source_frame（躯干）为原点，追踪 target_frames（各足部）。
2. transformer.data.target_pos_w 返回 (num_envs, num_feet, 3) 世界坐标张量。
3. 与 RayCaster 的区别：FrameTransformer 追踪关节末端位姿，不发射射线。

学习成果：
  运行后终端实时打印四个足部的世界坐标（Z 值约 0 表示贴地，大于 0 表示腾空）。

参考官方教程：scripts/tutorials/04_sensors/run_frame_transformer.py

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/04_go2_locomotion/step_15_2_frame_transformer.py

待实现：参照 scripts/tutorials/04_sensors/run_frame_transformer.py 完成本文件。
"""

# TODO: 参照 scripts/tutorials/04_sensors/run_frame_transformer.py 实现
pass
