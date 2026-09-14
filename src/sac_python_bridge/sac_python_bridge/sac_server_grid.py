#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from agricultural_msgs.msg import CoverageStatus
from agricultural_msgs.srv import SACPredict
from sensor_msgs.msg import LaserScan
import numpy as np
import torch as th
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import os
import threading
import math
import cv2

# ===== 全变分奖励工具函数 =====
def total_variation(img, img2=None, mode='non-iso'):
    """
    计算图像的全变分（Total Variation）
    
    参数：
        img: 输入图像数组 (2D)
        img2: 可选的第二张图像（通常是障碍物图）
        mode: 计算模式
            'sym-iso': 对称各向同性（最慢）
            'non-sym-iso': 非对称各向同性（中等）
            'non-iso': 非各向同性（最快，推荐用于奖励计算）
    
    返回：
        总变分值（标量）
    """
    assert mode in ['sym-iso', 'non-sym-iso', 'non-iso']
    img = img.astype(float)
    diff1 = np.abs(img[1:, :] - img[:-1, :])
    diff2 = np.abs(img[:, 1:] - img[:, :-1])
    if img2 is not None:
        assert img.shape == img2.shape
        img2 = np.maximum(img, img2)
        diff1 = np.minimum(diff1, np.abs(img2[1:, :] - img2[:-1, :]))
        diff2 = np.minimum(diff2, np.abs(img2[:, 1:] - img2[:, :-1]))
    if mode == 'sym-iso':
        tv = np.sum(np.sqrt(diff1[:, 1:] ** 2 +  diff2[1:, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, 1:] ** 2 +  diff2[:-1, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)) + \
             np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
        return tv / 4
    elif mode == 'non-sym-iso':
        return np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
    else:  # 'non-iso'
        return np.sum(diff1) + np.sum(diff2)

# ============================

class GridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=0), # 输入通道：4尺度 * 3层(覆盖+障碍+前沿)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with th.no_grad():
            sample_grid = th.as_tensor(observation_space.spaces['grid'].sample()[None]).float()
            n_flatten = self.cnn(sample_grid).shape[1]
            
        self.vector_net = nn.Sequential(
            nn.Linear(observation_space.spaces['vector'].shape[0], 64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        grid_features = self.cnn(observations['grid'])
        vector_features = self.vector_net(observations['vector'])
        combined = th.cat([grid_features, vector_features], dim=1)
        return self.fc(combined)

class SACServerGrid(Node):
    def __init__(self):
        super().__init__('sac_server_grid')
        
        try:
            self.declare_parameter('use_sim_time', True)
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        except Exception:
            pass
        try:
            self.declare_parameter('mode', 'train') # 'train' or 'inference'
            self.declare_parameter('use_gpu', True)
            self.declare_parameter('curriculum_name', '')  # 例如 lvl1_ep000
        except Exception:
            pass
        
        mode_param = self.get_parameter('mode').get_parameter_value().string_value
        self.mode = mode_param
        self.current_scan = None # 初始化雷达数据存储
        self.create_subscription(LaserScan, '/scan', self._laser_callback, 10)
        
        use_gpu_param = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.device = 'cuda' if (use_gpu_param and th.cuda.is_available()) else 'cpu'
        
        self.get_logger().info(f'🚀 核心升级：SAC Server 混合控制架构已启动 | Mode: {self.mode}')

        # ==== 重大改变：连续动作空间 ====
        # 输出一个 [dx, dy] 连续向量，取值范围 [-1.0, 1.0] 米
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 12 通道图像：4个尺度 (1x, 4x, 16x, 64x) * 每个尺度 3 个图层 (覆盖, 障碍, 前沿)
        # Vector 输入：8个原有特征 (相对前沿矢量、距离、覆盖率、避障雷达)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=255, shape=(12, 84, 84), dtype=np.uint8),
            'vector': spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        })

        # 【智能路径追踪】：即便是在 ROS2 install 目录下运行，也能自动找回 src 源码目录
        def find_models_dir():
            curr = os.path.dirname(os.path.abspath(__file__))
            # 向上溯源：从 site-packages 往回跳，直到找到包含 'src' 的文件夹
            for _ in range(12): 
                if os.path.isdir(os.path.join(curr, 'src')):
                    return os.path.join(curr, 'src', 'sac_python_bridge', 'models_and_logs')
                curr = os.path.dirname(curr)
            # 回落方案：如果没找到源码（比如二进制部署），则使用当前工程安装路径下的默认文件夹
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models_and_logs')

        base_save_dir = find_models_dir()
        os.makedirs(base_save_dir, exist_ok=True)
        
        # 模型和 buffer 始终在根目录，方便跨 Episode 连续加载
        self.model_path = os.path.join(base_save_dir, 'sac_model.zip')
        self.buffer_path = os.path.join(base_save_dir, 'latest_sac_buffer.pkl')
        
        # 日志按 level+ep 写入独立子目录
        curriculum_name = self.get_parameter('curriculum_name').get_parameter_value().string_value
        if curriculum_name:
            self.log_dir = os.path.join(base_save_dir, curriculum_name)
            self.get_logger().info(f'📁 课程训练模式：日志 → {self.log_dir} | 模型 → {base_save_dir}')
        else:
            self.log_dir = base_save_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 挂载最大熵连续控制器 SAC
        policy_kwargs = dict(
            features_extractor_class=GridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        # 使用 DummyEnv 满足 SB3 接口，避免 self 带来的 Node/Env 冲突
        class DummyEnv(gym.Env):
            def __init__(self, observation_space, action_space):
                self.observation_space = observation_space
                self.action_space = action_space
            def reset(self): return self.observation_space.sample()
            def step(self, action): return self.observation_space.sample(), 0.0, False, {}

        self.dummy_env = DummyEnv(self.observation_space, self.action_space)

        self.model = SAC(
            "MultiInputPolicy",
            env=self.dummy_env,
            learning_rate=1e-4, 
            buffer_size=100000,
            batch_size=128, 
            learning_starts=2000,
            gamma=0.99,
            ent_coef="auto", 
            target_entropy="auto",  
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=1,
            seed=30
        )
        
        from stable_baselines3.common.logger import configure
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)
        if hasattr(self.model, '_logger') is False:
            self.model._logger = new_logger

        if os.path.exists(self.model_path):
            try:
                self.model = SAC.load(self.model_path, env=self.dummy_env, device=self.device)
                self.model.set_logger(new_logger)
                if hasattr(self.model, '_logger') is False:
                    self.model._logger = new_logger
                self.get_logger().info(f'✅ 已加载 SAC 模型: {self.model_path}')
                
                if os.path.exists(self.buffer_path):
                    self.model.load_replay_buffer(self.buffer_path)
                    self.get_logger().info("🧠 Replay buffer loaded.")
                    
            except Exception as e:
                self.get_logger().error(f'加载失败: {e}')

        self.last_obs = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_grid_pos = None
        self.visited_cells = set()
        self.recent_positions = []
        # 如果加载了已有模型，继承它的历史步数；否则从0开始
        self.total_steps = self.model.num_timesteps if hasattr(self, 'model') else 0
        self.warmup_steps = 64
        self.train_lock = threading.Lock()
        self.is_running = True
        self.full_grid = None
        
        # 只有训练模式才启动后台线程
        if self.mode == 'train':
            self.train_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.train_thread.start()
        else:
            self.get_logger().info("ℹ️ 当前处于推理模式，后台训练线程已禁用。")
        
        # 使用并发回调组，防止训练阻塞服务响应
        self.cb_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.srv = self.create_service(SACPredict, 'sac_predict_service', self.sac_predict_callback, callback_group=self.cb_group)
        
        from std_srvs.srv import Empty
        self.reset_srv = self.create_service(Empty, 'reset_sac_state', self.reset_state_callback, callback_group=self.cb_group)
        
        self.res = 0.05  
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # 全变分（Total Variation）奖励参数
        # 【课程学习】：早期禁用TV，后期逐步启用，防止梯度压制
        # 当覆盖率 < tv_activation_threshold 时，TV奖励被禁用
        self.tv_activation_threshold = 20.0   # 覆盖率阈值 (20%)，超过此值才启用TV
        self.tv_coverage_ramp_factor = 1.0    # TV权重随覆盖率增加的倍增因子
        
        # 【局部TV奖励】：增量式计算，鼓励走直线、减少摇摆
        # 说明：路径平滑性在任何阶段都有价值，应该早期启用
        self.local_tv_reward_scale = 0.02     
        self.local_tv_reward_max = 0.3 
        
        # 【全局TV奖励】：根据覆盖区域的整体形状，鼓励规整的覆盖图案
        # 说明：全局形状约束在覆盖率较高时（>30%）才能真正起作用
        self.global_tv_reward_scale = 0.1     # 防止 TV 惩罚瞬间达到阈值
        self.global_tv_reward_max = 5.0       # 给予更大的浮动空间，防止锁死在 -1.0
        
        # 【TV计算配置】
        self.tv_mode = 'non-iso'              # 全变分计算模式：'sym-iso', 'non-sym-iso', 'non-iso'
        self.use_obstacles_in_tv = True       # 计算TV时是否考虑障碍物
        
        # 【TV状态跟踪】（每步更新）
        self.coverage_map = None              # 覆盖地图
        self.obstacle_map = None              # 障碍物地图
        self.global_total_variation = 0.0     # 全局TV累计值
        self.last_coverage_map = None         # 上一步的覆盖地图
        self.last_obstacle_map = None         # 上一步的障碍物地图
        self.last_coverage_pixels = 0         # 上一步的覆盖像素数，用于TV增量计算
        self.max_lin_vel = 0.5                # 最大线性速度（用于TV归一化）
        self.step_size = 0.1                  # 步长（秒，用于TV归一化）
        self.mower_radius = 0.5               # 覆盖半径（用于最大覆盖面积计算）
        self.pixels_per_meter = 20.0          # 像素/米（初始值，会动态更新）
        # =========================================


    def default_mode_train(self):
        return self.mode == 'train'

    def reset_state_callback(self, request, response):
        self.get_logger().info('🔄 Episode 重置，清除 SAC Last State Tracker')
        self.last_obs = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_grid_pos = None
        self.visited_cells.clear()
        self.recent_positions = []
        
        # 重置全变分相关的状态
        self.global_total_variation = 0.0
        self.last_coverage_map = None
        self.last_obstacle_map = None
        self.last_coverage_pixels = 0
        
        return response

    def calculate_reward(self, request, grid_pos, cell_value, coverage_percent, obs):
        """
        简约型理性奖励
        1. 割草奖：按全局像素增幅计算。
        2. 无效奖惩 (Waste Penalty)：如果不割新草，扣分幅度与“浪费”比例成正比。
        3. 转向/平滑罚：惩罚扭动，鼓励直线。
        4. 时间罚：固定的生存成本。
        """
        reward = 0.0
        r_x, r_y = grid_pos

        # A. 碰撞判定 (基于实时激光雷达锁存)
        # 使用在 _laser_callback 中锁存的碰撞状态，防止 Nav2 逃脱后漏判
        collision_detected = getattr(self, 'collision_latched', False)
        self.collision_latched = False  # 读取后立即清除，为下一步做准备

        if collision_detected:
            # 【关键修复】碰撞后必须立即返回！否则 reward 会被后面的正常计算覆盖
            # 同时更新覆盖计数，防止下一步的增量计算错乱
            grid_data = np.array(request.occupancy_grid.data, dtype=np.int8)
            self.last_global_coverage_sum = np.sum(grid_data == 50)
            self.reward_info = {
                "area": 0.0, "frontier_approach": 0.0,
                "turn": 0.0, "const": -10.0
            }
            return -10.0, 0.0
        grid_data = np.array(request.occupancy_grid.data, dtype=np.int8)
        curr_global_sum = np.sum(grid_data == 50)
        
        if not hasattr(self, 'last_global_coverage_sum') or self.last_global_coverage_sum is None:
            new_pixels = 0
        else:
            new_pixels = max(0, int(curr_global_sum - self.last_global_coverage_sum))
        
        self.last_global_coverage_sum = curr_global_sum

        # 1. 覆盖奖励 (reward_area)
        # 每像素面积为 0.05 * 0.05 = 0.0025 平方米
        # 设定：覆盖 1 平方米 (400像素) 奖励 +5.0
        # 因此每像素的奖励权重为 5.0 / 400 = 0.0125
        reward_area = new_pixels * 0.0125

        # 2. 边缘探索奖励
        # 核心思想：只奖励"靠近 frontier 的变化量"，杜绝"站在原地蹭分"
        # 使用真实米制距离，移动 1 米 → delta = 1.0，信号不被归一化压缩
        frontier_dist_m = getattr(self, 'frontier_dist_meters', 0.0)
        
        if not hasattr(self, 'last_frontier_dist_m') or self.last_frontier_dist_m is None:
            self.last_frontier_dist_m = frontier_dist_m
        
        # delta > 0 表示靠近了 frontier，delta < 0 表示远离了
        delta_m = self.last_frontier_dist_m - frontier_dist_m
        reward_frontier_approach = delta_m * 0.5  # 靠近 1 米 → +0.5 奖励
        self.last_frontier_dist_m = frontier_dist_m
        
        # 3. 高覆盖率阶段放大引导信号（封顶 3.0，防止噪声被过度放大）
        coverage_urgency = min(5.0, 1.0 + max(0.0, coverage_percent - 50.0) * 0.1)

        # 4. 转向惩罚与无覆盖怠工重罚 (Waste & Laziness Penalty)
        reward_turn = 0.0
        
        if self.last_action is not None:
            v_x = float(self.last_action[0])
            v_w = abs(float(self.last_action[1]))

            if new_pixels == 0:
                if v_x < 0.2 and v_w > 0.1:
                    reward_turn = -1.2 * v_w - 0.3 * (1.0 - v_x)
            else:
                reward_turn = -0.8 * abs(v_w) 
                
            self.prev_action = np.copy(self.last_action)

        # 5. 步幅距离惩罚 (Time & Distance Penalty)
        # 引入动作幅值 (action_norm) 惩罚
        base_penalty = -0.05
        if self.last_action is not None:
            # 使用 L2 平方惩罚动作边界（对标 user 建议），强烈惩罚饱和区动作
            action_l2_sq = float(self.last_action[0])**2 + float(self.last_action[1])**2
            reward_const = base_penalty - 0.1 * action_l2_sq
        else:
            reward_const = base_penalty

        # 6. 【全变分（Total Variation）奖励】
        # 关键修复：(1) 使用参数化权重替换硬编码值 (2) 添加完整的global TV (3) 物理单位归一化
        reward_itv = 0.0   # 局部平滑性（增量式）
        reward_gtv = 0.0   # 全局形状（整体规整度）
        
        if self.full_grid is not None and self.coverage_map is not None and not collision_detected:
            coverage_layer = (self.coverage_map > 0).astype(float)
            obstacle_layer = (self.obstacle_map > 0).astype(float) if self.use_obstacles_in_tv else None
            
            # 计算当前的全变分值
            if obstacle_layer is not None:
                total_variation_new = total_variation(coverage_layer, obstacle_layer, mode=self.tv_mode)
            else:
                total_variation_new = total_variation(coverage_layer, mode=self.tv_mode)
            
            coverage_pixels = np.sum(coverage_layer > 0)
            
            # ===== A. 局部TV奖励（Local Incremental TV） - 鼓励路径平滑 =====
            # 【对标 RL-cpp】reward_itv = -total_variation_diff * self.local_tv_reward_scale / 2.5
            if self.last_coverage_map is not None:
                if obstacle_layer is not None and self.last_obstacle_map is not None:
                    total_variation_old = total_variation(self.last_coverage_map, self.last_obstacle_map, mode=self.tv_mode)
                else:
                    total_variation_old = total_variation(self.last_coverage_map, mode=self.tv_mode)
                total_variation_diff = total_variation_new - total_variation_old
            else:
                total_variation_diff = 0.0
            
            if total_variation_diff != 0:
                # 只惩罚TV增加（边缘碎片化），不奖励TV减少（被覆盖奖励处理）
                reward_itv = -total_variation_diff * self.local_tv_reward_scale / 2.5
                reward_itv = np.sign(reward_itv) * min(abs(reward_itv), self.local_tv_reward_max)
            
            # ===== B. 全局TV奖励（Global TV） - 鼓励覆盖区域形状规整 ====
            if coverage_pixels > 0:
                # 更新全局TV累计值
                self.global_total_variation = total_variation_new
                
                # 圆形的周长/sqrt(面积) ≈ 2*sqrt(π) ≈ 3.54，但由于网格离散化，用4更准确
                normalization_factor = np.sqrt(coverage_pixels) if coverage_pixels > 0 else 1.0
                reward_gtv = -self.global_total_variation / normalization_factor / 4.0
                
                # 根据覆盖率启用课程学习
                # 只在覆盖率足够高时（>20%）启用全局约束，防止早期过度惩罚
                if coverage_percent < self.tv_activation_threshold:
                    reward_gtv *= 0.1  # 早期大幅削弱
                else:
                    # 随覆盖率提高，逐步增强TV约束
                    coverage_ramp = 1.0 + (coverage_percent - self.tv_activation_threshold) * self.tv_coverage_ramp_factor * 0.1
                    reward_gtv *= coverage_ramp
                
                # 乘以缩放因子并应用截断
                reward_gtv *= self.global_tv_reward_scale
                reward_gtv = np.sign(reward_gtv) * min(abs(reward_gtv), self.global_tv_reward_max)
            
            # 保存当前地图以供下一步使用
            self.last_coverage_map = coverage_layer.copy()
            self.last_coverage_pixels = coverage_pixels if coverage_pixels > 0 else 0
            if self.use_obstacles_in_tv and self.obstacle_map is not None:
                self.last_obstacle_map = obstacle_layer.copy()

        reward_tv = reward_itv + reward_gtv

        # --- 最终汇总（加入全变分奖励）---
        reward = (reward_area +                                      # 覆盖奖励
                 reward_frontier_approach * coverage_urgency +       # 势能差 × 紧迫度（最大 10x）
                 reward_turn +                                       # 转向惩罚
                 reward_tv +                                         # 全变分奖励（局部+全局）
                 reward_const)                                       # 时间成本
        
        self.reward_info = {
            "area": reward_area,
            "frontier_approach": reward_frontier_approach,
            "turn": reward_turn,
            "tv_local": reward_itv,           # 局部平滑性 (对应公式里的 local incremental TV)
            "tv_global": reward_gtv,          # 全局形状 (对应公式里的 global shape TV)
            "const": reward_const
        }

        # 补充：雷达有效性检查日志 (排查为何雷达没效果)
        if not hasattr(self, '_last_laser_time'):
            self._last_laser_time = self.get_clock().now()
            self._laser_count = 0
            
        if self.current_scan is not None:
            now = self.get_clock().now()
            if (now - self._last_laser_time).nanoseconds > 1e9: # 每秒检查一次
                if hasattr(self, '_laser_count') and self._laser_count == 0:
                    self.get_logger().warn("💣 警告：/scan 话题已超时或无数据接收！请检查仿真环境。")
                self._last_laser_time = now
                self._laser_count = 0
        
        return float(np.nan_to_num(reward, nan=-1.0, posinf=0.0, neginf=-1.0)), 0.0

    def _compute_frontier_map(self, coverage, obstacles, free_mask=None):
        """
        边缘检测 V3.1：排除墙壁膨胀区 + 排除地图边界外的虚无像素
        free_mask: 布尔数组，True = 原始地图中确认的自由空间 (map_grid == 0)
        """
        import cv2
        kernel = np.ones((3,3), np.uint8)
        dilated_cov = cv2.dilate(coverage, kernel, iterations=1)
        # 恢复 10px = 0.5m 的专家安全区，确保目标点在 Nav2 的致命碰撞区之外
        obstacle_dilated = cv2.dilate(obstacles, kernel, iterations=10)
        frontier = np.logical_and(dilated_cov > 0, coverage == 0)
        frontier = np.logical_and(frontier, obstacle_dilated == 0)
        # 关键：只保留原始地图中确认为自由空间的像素，排除边界外的默认值=0区域
        if free_mask is not None:
            frontier = np.logical_and(frontier, free_mask)
        return frontier.astype(np.uint8)

    def _get_rotated_multiscale_window(self, rx, ry, ryaw, frontier_full, input_size=84):
        """ 1:1 复刻 rl-cpp 的 _get_relative_map 高性能逻辑 """
        import cv2
        # 基础坐标
        ri = (rx - self.origin_x) / self.res
        rj = (ry - self.origin_y) / self.res
        angle_deg = math.degrees(ryaw)
        
        scales = [1, 4, 16, 64]
        results = []
        
        # 全局尺寸
        h, w = self.full_grid.shape[1], self.full_grid.shape[2]
        
        for n, s in enumerate(scales):
            # 1. 先通过 INTER_AREA 预缩放，减少 warpAffine 负担并防止混叠
            # 缩放比例 sc 需要考虑地图实际大小，模拟 sc = min(scale, self.size_p)
            sc = s
            # 计算缩放后的全局图尺寸
            new_w = int(0.5 + w / sc)
            new_h = int(0.5 + h / sc)
            
            # 使用不同插值方式处理连贯层与二值层
            res_win0 = cv2.resize(self.full_grid[0], (new_w, new_h), interpolation=cv2.INTER_AREA)
            res_win1 = cv2.resize(self.full_grid[1], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            res_win2 = cv2.resize(frontier_full, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 2. 计算在缩放后的图上的变换矩阵
            # 机器人坐标也需要对应缩放
            ri_sc = ri / sc
            rj_sc = rj / sc
            
            # 仿射变换三部曲：
            # T1: 把机器人移到原点
            T1 = np.eye(3)
            T1[0, 2] = -ri_sc
            T1[1, 2] = -rj_sc
            
            # R: 旋转 (Forward is Up, OpenCV rotate is CCW)
            R = np.eye(3)
            R[:2] = cv2.getRotationMatrix2D((0, 0), 90 - angle_deg, 1.0)
            
            # T2: 把原点移到观测窗中心
            T2 = np.eye(3)
            T2[0, 2] = input_size / 2
            T2[1, 2] = input_size / 2
            
            M_final = (T2 @ R @ T1)[:2]
            
            # 3. 在极小图像上执行最终的变换
            # 使用 BORDER_CONSTANT 补齐边界
            obs_win0 = cv2.warpAffine(res_win0, M_final, (input_size, input_size), 
                                     flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            obs_win1 = cv2.warpAffine(res_win1, M_final, (input_size, input_size), 
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            obs_win2 = cv2.warpAffine(res_win2, M_final, (input_size, input_size), 
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            obs_win2 = (obs_win2 > 0).astype(np.uint8) * 255
            
            results.append(obs_win0)
            results.append(obs_win1)
            results.append(obs_win2)
            
        res_array = np.stack(results).astype(np.uint8)
        
        # ======将CNN视野输出为直观图片======
        if not hasattr(self, '_debug_save_count'):
            self._debug_save_count = 0
            
        # 每隔20步保存一次，避免过度占用硬盘资源
        if self._debug_save_count % 20 == 0:
            import os
            # Row 1: Coverage maps (1x, 4x, 16x, 64x)
            row1 = np.hstack([res_array[0], res_array[3], res_array[6], res_array[9]])
            # Row 2: Obstacle maps (1x, 4x, 16x, 64x)
            row2 = np.hstack([res_array[1], res_array[4], res_array[7], res_array[10]])
            # Row 3: Frontier maps (1x, 4x, 16x, 64x) - 新增
            row3 = np.hstack([res_array[2], res_array[5], res_array[8], res_array[11]])
            
            # 合成并采用最近邻插值放大两倍，便于肉眼观察像素晶格
            debug_img = np.vstack([row1, row2, row3])
            # 画布宽度 84*4=336, 高度 84*3=252
            debug_img_enlarged = cv2.resize(debug_img, (336 * 2, 252 * 2), interpolation=cv2.INTER_NEAREST)
            
            # 直接保存到AI的当前会话目录供调用展示
            save_path = '/home/cmh/MyProject/CNN_Input/cnn_input_visual.png'
            cv2.imwrite(save_path, debug_img_enlarged)
            
        self._debug_save_count += 1
        # ================================================
            
        return res_array

    def extract_state(self, request):
        grid_data = np.array(request.occupancy_grid.data, dtype=np.int8)
        w = request.occupancy_grid.info.width
        h = request.occupancy_grid.info.height
        res = request.occupancy_grid.info.resolution
        origin_x = request.occupancy_grid.info.origin.position.x
        origin_y = request.occupancy_grid.info.origin.position.y
        self.res, self.origin_x, self.origin_y = res, origin_x, origin_y
        
        map_grid = grid_data.reshape((h, w))
        
        # 预分配/更新全地图图层
        coverage_layer = np.zeros_like(map_grid, dtype=np.uint8)
        obstacle_layer = np.zeros_like(map_grid, dtype=np.uint8)
        
        coverage_layer[map_grid == 50] = 127 # 已覆盖
        obstacle_layer[map_grid == 100] = 255 # 障碍物
        self.full_grid = np.stack([coverage_layer, obstacle_layer])
        # 保存「已知自由空间」掩码：只有 map_grid==0 的像素才是地图内真正的未覆盖自由区
        # 排除边界外默认填充的 0 值、未知区(-1)、障碍物(100)
        self._free_space_mask = (map_grid == 0)

        # 获取机器人位姿
        rx = request.robot_pose.pose.position.x
        ry = request.robot_pose.pose.position.y
        q = request.robot_pose.pose.orientation
        ryaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        # 动态获取地图属性，确保坐标系对齐
        res = request.occupancy_grid.info.resolution
        origin_x = request.occupancy_grid.info.origin.position.x
        origin_y = request.occupancy_grid.info.origin.position.y
        self.res, self.origin_x, self.origin_y = res, origin_x, origin_y

        gx_clamped = np.clip(int((rx - origin_x) / res), 0, w - 1)
        gy_clamped = np.clip(int((ry - origin_y) / res), 0, h - 1)
        ri = gx_clamped  # 机器人在栅格中的像素 X 坐标
        rj = gy_clamped  # 机器人在栅格中的像素 Y 坐标
        cell_val = map_grid[gy_clamped, gx_clamped]

        # 每步实时更新 + free_mask 排除地图外虚无区域
        # 先计算前沿图，供后续 CNN 输入和向量计算使用
        frontier_full = self._compute_frontier_map(
            self.full_grid[0], self.full_grid[1],
            free_mask=getattr(self, '_free_space_mask', None)
        )

        # 多尺度 ego-centric 获取
        grid_obs = self._get_rotated_multiscale_window(rx, ry, ryaw, frontier_full, input_size=84)

        y_coords, x_coords = np.where(frontier_full == 1)
        if len(y_coords) > 0:
            dists = (x_coords - ri)**2 + (y_coords - rj)**2
            min_idx = np.argmin(dists)
            self._cached_frontier_target = (x_coords[min_idx], y_coords[min_idx])
        else:
            self._cached_frontier_target = None

        if self._cached_frontier_target is not None:
            tx, ty = self._cached_frontier_target
            dx = tx - ri
            dy = ty - rj
            nearest_zero_dist = math.sqrt(dx*dx + dy*dy)
            
            cos_y = math.cos(ryaw)
            sin_y = math.sin(ryaw)
            lx = dx * cos_y + dy * sin_y
            ly = -dx * sin_y + dy * cos_y
            
            n = math.sqrt(lx*lx + ly*ly) if (lx!=0 or ly!=0) else 1.0
            vec_to_zero = [lx/n, ly/n]
        else:
            # 如果全局都没有边缘（即覆盖完成），向量指示停止或随机探索
            vec_to_zero = [0.0, 0.0]
            nearest_zero_dist = 0.0

        # 【can_move】：基于机器人姿态的前、左、后、右真实射线探测
        # 探索距离为 12 像素（约 0.6米，略大于机器人的 0.5m 碰撞半径）
        check_dist = 12
        can_move = []
        directions = [0.0, math.pi/2, math.pi, -math.pi/2] # 前, 左, 后, 右
        
        for d_yaw in directions:
            abs_yaw = ryaw + d_yaw
            dx = math.cos(abs_yaw)
            dy = math.sin(abs_yaw)
            is_clear = True
            # 发射射线检查这 0.6 米内是否有碰撞风险 (100) 或地图越界
            for step in range(3, check_dist): 
                nx = int(gx_clamped + dx * step)
                ny = int(gy_clamped + dy * step)
                if ny < 0 or ny >= h or nx < 0 or nx >= w or map_grid[ny, nx] == 100:
                    is_clear = False
                    break
            can_move.append(1.0 if is_clear else -1.0)
        cov_pct = request.coverage_status.coverage_percentage / 100.0
        norm_dist = min(nearest_zero_dist / float(max(w, h)), 1.0)
        # 保存真实米制距离，供前沿引导奖励使用
        self.frontier_dist_meters = nearest_zero_dist * res
        
        vector_state = np.array([
            vec_to_zero[0], vec_to_zero[1], norm_dist, cov_pct,
            can_move[0], can_move[1], can_move[2], can_move[3]
        ], dtype=np.float32)

        # 保存覆盖地图和障碍物地图以供TV奖励计算
        self.coverage_map = coverage_layer.copy()
        self.obstacle_map = obstacle_layer.copy()
        # 更新像素/米的比例（用于TV奖励的归一化）
        self.pixels_per_meter = 1.0 / res if res > 0 else 20.0

        return {
            'grid': grid_obs,
            'vector': vector_state
        }, (gx_clamped, gy_clamped), cell_val, rx, ry, ryaw

    def _training_loop(self):
        """ 后台训练循环：在不阻塞机器人的情况下执行梯度下降 """
        self.get_logger().info("⚙️ SAC 后台训练线程已启动 (4-Channel Scale Input)...")
        import time
        self.total_train_steps = getattr(self, 'total_steps', 0)
        while self.is_running:
            if self.model.replay_buffer.size() >= self.model.batch_size and self.default_mode_train():
                with self.train_lock:
                    cov_pct = getattr(self, 'current_coverage_percent', 0.0)
                    
                    if cov_pct < 70.0:
                        current_ent = 0.3
                    elif cov_pct > 95.0:
                        current_ent = 0.05
                    else:
                        progress = (cov_pct - 70.0) / 25.0
                        current_ent = 0.05 + (0.3 - 0.05) * 0.5 * (1.0 + math.cos(math.pi * progress))
                    current_lr = 1e-4 
                    
                    if not hasattr(self.model, '_custom_lr_patched'):
                        self.model._update_learning_rate = lambda *args, **kwargs: None
                        self.model._custom_lr_patched = True
                    
                    # 1. 退火 Actor 和 Critic 的学习率
                    for param_group in self.model.actor.optimizer.param_groups:
                        param_group['lr'] = current_lr
                    for param_group in self.model.critic.optimizer.param_groups:
                        param_group['lr'] = current_lr
                        
                    if hasattr(self.model, 'ent_coef_optimizer') and self.model.ent_coef_optimizer is not None:
                        import torch
                        # 停止优化器的更新
                        for param_group in self.model.ent_coef_optimizer.param_groups:
                            param_group['lr'] = 0.0
                        # 直接设定底层 tensor 值
                        self.model.log_ent_coef.data = torch.log(torch.tensor(current_ent, device=self.model.device).float())

                    # --- 执行网络参数更新 ---
                    self.model.train(gradient_steps=1, batch_size=self.model.batch_size)
                    self.total_train_steps += 1
                    
                    # 周期性保存模型
                    if self.total_train_steps % 500 == 0:
                        self.model.save(self.model_path)
                        self.model.save_replay_buffer(self.buffer_path)
                        self.get_logger().info(f"💾 Model & Buffer saved to {os.path.dirname(self.model_path)}")
                    
                    if self.total_train_steps % 5000 == 0:
                        checkpoint = os.path.join(os.path.dirname(self.model_path), f"SAC_mower_{self.total_train_steps}.zip")
                        self.model.save(checkpoint)
                time.sleep(0.01) 
            else:
                time.sleep(1.0) 
                
    def _laser_callback(self, msg):
        """ 接收激光雷达数据，保留有效的非无穷大距离值 """
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.current_scan = ranges
        
        # 如果检测到碰撞，将其锁存，直到下一个 step 开始时作为负反馈发出
        if np.min(ranges) < 0.52:
            self.collision_latched = True
        
        # 增加计数器，用于验证话题活跃度
        if not hasattr(self, '_laser_count'): self._laser_count = 0
        self._laser_count += 1

    def sac_predict_callback(self, request, response):
        is_terminal = False
        try:
            # 接口适配：解包 6 个返回值
            obs, grid_pos, cell_val, rx, ry, ryaw = self.extract_state(request)
            coverage = request.coverage_status.coverage_percentage
            # 实时记录覆盖率，供后台退火线程使用
            self.current_coverage_percent = coverage
            is_terminal = (coverage >= 100.0)

            # RL Phase: 数据存入 Replay Buffer
            if self.last_obs is not None and self.default_mode_train():
                # 接口适配：解包 2 个返回值
                reward, curr_tv = self.calculate_reward(request, grid_pos, cell_val, coverage, obs)
                if is_terminal:
                    reward += 1000.0 if coverage >= 100.0 else 0.0

                with self.train_lock:
                    if request.coverage_status.repeated_coverage_count == 1:
                        expert_action = np.array([request.robot_velocity.linear.x, request.robot_velocity.linear.y], dtype=np.float32)
                        penalty_reward = -2.0
                        self.model.replay_buffer.add(
                            self.last_obs, obs,
                            self.last_action, np.array([penalty_reward]),
                            np.array([is_terminal]), [{}]
                        )
                        self.model.replay_buffer.add(
                            self.last_obs, obs,
                            expert_action, np.array([reward]),
                            np.array([is_terminal]), [{}]
                        )
                    else:
                        self.model.replay_buffer.add(
                            self.last_obs, obs,
                            self.last_action, np.array([reward]),
                            np.array([is_terminal]), [{}]
                        )

            # Inference: SAC produces the continuous [dx, dy] coordinate displacement
            self.total_steps += 1
            
            # 将关键指标注入日志
            if self.last_obs is not None:
                self.model.logger.record("coverage", float(coverage))
                self.model.logger.record("total_reward", float(reward if 'reward' in locals() else 0.0))
                if hasattr(self, 'reward_info'):
                    for key, val in self.reward_info.items():
                        self.model.logger.record(f"reward_{key}", float(val))
                
                q1, q2 = 0.0, 0.0
                if self.total_steps > self.warmup_steps:
                    try:
                        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                        with th.no_grad():
                            actor_action, log_prob = self.model.policy.actor.action_log_prob(obs_tensor)
                            q_vals = self.model.critic(obs_tensor, actor_action)
                            q1 = q_vals[0].mean().item()
                            q2 = q_vals[1].mean().item()
                    except Exception:
                        pass

                # 每一位步都执行 dump，确保 CSV 和 TensorBoard 能够捕捉到每一个动作的反馈
                if self.total_steps % 1 == 0:
                    self.model.logger.dump(step=self.total_steps)

            if self.default_mode_train() and self.total_steps <= self.warmup_steps:
                vx, vy = obs['vector'][0], obs['vector'][1]
                dist = np.hypot(vx, vy)
                if dist > 0.01:
                    # 借用向量方向，加入 0.2 的噪声进行探索
                    action = np.array([vx/dist, vy/dist], dtype=np.float32)
                    action += np.random.normal(0, 0.2, size=2).astype(np.float32)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = self.action_space.sample()
                
                if self.total_steps % 10 == 0:
                    self.get_logger().info(f'🔥 引导预热阶段 [{self.total_steps}/{self.warmup_steps}]')
            else:
                with self.train_lock:
                    action, _ = self.model.predict(obs, deterministic=(not self.default_mode_train()))
                if self.total_steps == self.warmup_steps + 1:
                    self.get_logger().info(f'🧠 预热结束！SAC 已经接手。buffer={self.model.replay_buffer.size()}')

            self.last_obs = obs
            self.last_action = action
            self.last_grid_pos = grid_pos
            self.visited_cells.add(grid_pos)

            response.action = 0
            response.q_values = [float(action[0]), float(action[1])]
            response.confidence = 1.0
            response.success = True
        except Exception as e:
            self.get_logger().error(f'❌ SAC predict 回调异常: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            response.action = 0
            response.q_values = [float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1))]
            response.confidence = 0.0
            response.success = False

        return response

def main(args=None):
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init(args=args)
    node = SACServerGrid()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.mode == 'train':
            node.model.save(node.model_path)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
