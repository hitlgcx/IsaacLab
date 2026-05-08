"""
Step 8.0: PPO 算法配置与神经网络定义
================================================================
文件描述：脱离 Isaac Sim，纯 Python 环境下定义 Actor-Critic 网络结构和 PPO 超参数。

剥洋葱重点：
1. Policy（Actor）使用 GaussianMixin：输出动作均值 + 可学习的标准差，用于随机探索。
2. Value（Critic）使用 DeterministicMixin：输出单个标量作为状态价值估计。
3. PPO_CFG 字典是训练大纲，Runner 会自动解析并搭建完整训练流水线。

学习成果：
  运行后终端打印 Actor 和 Critic 的网络结构，无需 Isaac Sim 即可单独运行。

运行方式：
  conda activate env_isaaclab
  python scripts/my_learning/03_skrl_training/step_8_0_algo_cfg.py
"""

import torch
import torch.nn as nn
import gymnasium.spaces as spaces

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

# ==========================================
# 1. 定义神经网络 (运动员的大脑)
# ==========================================

class Policy(GaussianMixin, Model):
    """
    策略网络 (Actor)：负责“看”和“动”
    输入: obs (当前状态) -> 输出: action (应该施加多大的力)
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions)

        # 两层 32 个神经元的全连接层 (MLP)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, self.num_actions)
        )
        # 动作的标准差 (控制探索的随机性程度)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # inputs["states"] 存放的就是我们的 obs 张量
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    """
    价值网络 (Critic)：负责“打分”
    输入: obs (当前状态) -> 输出: value (预测这局游戏未来还能拿多少分)
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1) # 价值网络只输出 1 个评分
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ==========================================
# 2. 定义训练超参数 (教练的训练大纲)
# ==========================================

# 这是一个标准的 PPO 算法配置字典
PPO_CFG = {
    "rollouts": 16,               # 每次网络更新前，控制小车走多少步
    "learning_rate": 1e-3,        # 学习率
    "learning_rate_scheduler": "KLAdaptiveLR",
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
    "discount_factor": 0.99,      # Gamma
    "gae_lambda": 0.95,            # GAE lambda
    "grad_norm_clip": 1.0,        
    "ratio_clip": 0.2,            
    "value_clip": 0.2,
    "entropy_loss_scale": 0.0,    
    "value_loss_scale": 2.0,      
    "experiment": {
        "directory": "logs/skrl",          
        "experiment_name": "cartpole_ppo", 
        "write_interval": 16,              
        "checkpoint_interval": 160,        
        "wandb": False                     
    }
}

# ==========================================
# 3. 验证区
# ==========================================

def main():
    print("="*50)
    print("[INFO] 教练的训练计划 (PPO_CFG) 已就绪！")
    print(f"       每次更新采集步数: {PPO_CFG['rollouts']}")
    print(f"       学习率: {PPO_CFG['learning_rate']}")
    print(f"       日志目录: {PPO_CFG['experiment']['directory']}/{PPO_CFG['experiment']['experiment_name']}")
    print("="*50 + "\n")
    
    # 模拟环境的维度空间
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    
    # 实例化大脑
    actor = Policy(obs_space, act_space, "cpu")
    critic = Value(obs_space, act_space, "cpu")
    
    print("[INFO] 运动员的大脑 (网络结构) 创建成功：")
    print("Actor (动作策略) 网络结构:")
    print(actor.net)
    print("\nCritic (价值评估) 网络结构:")
    print(critic.net)
    print("="*50)

if __name__ == "__main__":
    main()