#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from agricultural_msgs.msg import CoverageStatus
from agricultural_msgs.srv import SACPredict
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf2_ros import TransformListener, Buffer
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import time
import math
import threading
import json
import os

try:
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
except ImportError:
    print("未安装 cv2 模块，将影响核心切分算法。请运行: pip install opencv-python")

class SACTrainerClient(Node):
    def __init__(self):
        super().__init__('sac_hybrid_trainer_client')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('json_path', '')
        self.declare_parameter('mode', 'inference') # 'train' or 'inference'
        self.declare_parameter('skip_phase1', False) # True means skip geometric boustrophedon
        self.declare_parameter('stop_after_phase1', False) # True means stop after Phase 1 (for pure_boustrophedon baseline)
        self.declare_parameter('use_heuristic', False) # True means always use greedy frontier instead of SAC
        self.declare_parameter('target_exit_coverage', 98.0) # Used to finish training loop
        
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        
        self.coverage_percentage = 0.0
        self.static_map_data = None 

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CoverageStatus, 'coverage_status', self.status_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        
        self.grid_pub = self.create_publisher(OccupancyGrid, 'covered_grid', map_qos)
        self.status_pub = self.create_publisher(CoverageStatus, 'coverage_status', 10)
        self.hud_pub = self.create_publisher(Image, 'hms_hud', 10)
        
        stop_p1 = self.get_parameter('stop_after_phase1').get_parameter_value().bool_value
        skip_p1 = self.get_parameter('skip_phase1').get_parameter_value().bool_value
        use_heu = self.get_parameter('use_heuristic').get_parameter_value().bool_value
        
        if stop_p1:
            self.alg_name = "Pure Boustrophedon (PBC)"
        elif skip_p1 and use_heu:
            self.alg_name = "Frontier-based (FCE)"
        elif skip_p1 and not use_heu:
            self.alg_name = "Pure SAC"
        else:
            self.alg_name = "HMS-CPP Algorithm (Ours)"
            
        self.current_status = "INITIALIZING"

        try:
            self.cv_bridge = CvBridge()
        except:
            self.cv_bridge = None
        self.hud_timer = self.create_timer(0.2, self.publish_hud)
        self.episode_start_time = None
        self.current_phase = "Initializing..."

        
        self.res = None
        self.width_m = None
        self.height_m = None
        self.origin_x = None
        self.origin_y = None

        # 推导 progress.csv 日志路径，以便 Phase 1 也能将覆盖率写入统一日志
        json_path_str = self.get_parameter('json_path').value
        self.progress_csv_path = None
        if json_path_str:
            cur_name = os.path.basename(json_path_str).split('.')[0]
            curr = os.path.dirname(os.path.abspath(__file__))
            base_dir = None
            for _ in range(12): 
                if os.path.isdir(os.path.join(curr, 'src')):
                    base_dir = os.path.join(curr, 'src', 'sac_python_bridge', 'models_and_logs')
                    break
                curr = os.path.dirname(curr)
            if not base_dir:
                base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models_and_logs')
            self.progress_csv_path = os.path.join(base_dir, cur_name, 'phase1_progress.csv')

            try:
                os.makedirs(os.path.dirname(self.progress_csv_path), exist_ok=True)
                with open(self.progress_csv_path, 'w') as f:
                    f.write("total_reward,coverage\n")
            except Exception:
                pass
        self.cols = None
        self.rows = None
        self.grid_data = None
        

        self.last_actual_fwd = 0.0
        self.last_actual_left = 0.0
        self.action_was_overridden = False
        
        self.work_radius_cells = 13 # 工作半径
        
        self.sac_client = self.create_client(SACPredict, 'sac_predict_service')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.target_coverage = self.get_parameter('target_exit_coverage').get_parameter_value().double_value
        self.step_size = 1.0  

        self.path_pub = self.create_publisher(Path, 'robot_trajectory', 10)
        self.trajectory_poses = []
        self.last_trajectory_pose = None
        self.total_path_length = 0.0
        
        self.home_pose = None

        json_path = self.get_parameter('json_path').value
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    pos, yaw = data['robot_pose']
                    self.home_pose_x = pos[0]
                    self.home_pose_y = pos[1]
                    self.home_pose_yaw = yaw
                    self.get_logger().info(f"📍 训练引擎成功解析专属微环境 JSON: 起点固定在 X={pos[0]:.2f}, Y={pos[1]:.2f}")
            except Exception as e:
                self.get_logger().error(f"JSON 读取失败: {e}")
        else:
            self.get_logger().info(f"⚠️ 未指定 JSON 路径，将从 TF 树动态捕获起步点作为永久安全基地")

        self._last_grid_publish_time = 0.0  # 限制 Grid 发布频率，防止 RViz 长时间运行崩溃
        self.create_timer(0.05, self.maintenance_loop)
    
    def maintenance_loop(self):
        pose = self.get_current_pose()
        if pose:
            if getattr(self, 'home_pose_x', None) is None:
                self.home_pose_x = pose.pose.position.x
                self.home_pose_y = pose.pose.position.y
                self.home_pose_yaw = 0.0 # simplified
                self.get_logger().info(f"🏠 已锁定此时小车所在的坐标为之后复活的安全大本营! ({self.home_pose_x:.2f}, {self.home_pose_y:.2f})")
                
            self.update_coverage(pose.pose.position.x, pose.pose.position.y)
            self.update_trajectory(pose)
        self.publish_grid()
        self.publish_trajectory()

    def update_trajectory(self, pose):
        if self.last_trajectory_pose is None:
            self.trajectory_poses.append(pose)
            self.last_trajectory_pose = pose
            return
        dx = pose.pose.position.x - self.last_trajectory_pose.pose.position.x
        dy = pose.pose.position.y - self.last_trajectory_pose.pose.position.y
        dist_sq = dx * dx + dy * dy
        if dist_sq > 4.0:
            self.trajectory_poses = [pose]
            self.last_trajectory_pose = pose
        elif dist_sq > 0.01:
            self.total_path_length += math.sqrt(dist_sq)
            self.trajectory_poses.append(pose)
            self.last_trajectory_pose = pose

    def publish_trajectory(self):
        if not hasattr(self, '_last_path_publish_time'):
            self._last_path_publish_time = 0.0
        
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_path_publish_time < 1.0:
            return  # 限制到 1Hz，防止长路径数组把 RViz 内存卡爆
        self._last_path_publish_time = now

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        path_msg.poses = self.trajectory_poses
        self.path_pub.publish(path_msg)

    def reset_trajectory(self):
        self.trajectory_poses = []
        self.last_trajectory_pose = None

    def status_callback(self, msg):
        self.coverage_percentage = msg.coverage_percentage

    def xyy_to_pose(self, x, y, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        # 用基本的四元数公式
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception:
            return None

    def update_coverage(self, x, y):
        if self.res is None or self.grid_data is None: return
        mx = int((x - self.origin_x) / self.res)
        my = int((y - self.origin_y) / self.res)
        for dx in range(-self.work_radius_cells, self.work_radius_cells + 1):
            for dy in range(-self.work_radius_cells, self.work_radius_cells + 1):
                cx, cy = mx + dx, my + dy
                if 0 <= cx < self.cols and 0 <= cy < self.rows:
                    self.grid_data[cy, cx] = 50

    def publish_grid(self):
        if self.grid_data is None: return
        
        now = time.time()
        if now - self._last_grid_publish_time < 0.5:
            return
        self._last_grid_publish_time = now
        
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.res
        msg.info.width = self.cols
        msg.info.height = self.rows
        msg.info.origin.position.x = float(self.origin_x)
        msg.info.origin.position.y = float(self.origin_y)
        # 使用原始地图的旋转角，确保两层地图完美叠加
        if hasattr(self, 'origin_pose'):
            msg.info.origin.orientation = self.origin_pose.orientation
        else:
            msg.info.origin.orientation.w = 1.0  # 保底合法值
        msg.data = self.grid_data.flatten().tolist()
        self.grid_pub.publish(msg)

        if self.static_map_data is not None:
            import cv2

            if not hasattr(self, '_reachable_mask') or self._reachable_mask is None:
                walls = np.logical_or(self.static_map_data == 100, self.static_map_data == -1).astype(np.uint8)
                # 形态学闭运算：填补障碍物之间的窄缝（与目标点选择逻辑一致）
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, close_kernel)
                # 安全膨胀（iterations=10 ≈ 0.5m），用于目标点选择
                inflated_walls = cv2.dilate(walls_closed, np.ones((3,3), np.uint8), iterations=10)
                self._reachable_mask = (inflated_walls == 0)
                # 1. 以机器人底盘可达区为起点，膨胀 work_radius_cells（刷子半径）
                # 2. 与非障碍物区域取交集 → 刷子能扫到的所有自由像素
                # 这样锐角深处、柱子背面等刷子物理上够不到的死角会被排除出分母
                brush_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * self.work_radius_cells + 1, 2 * self.work_radius_cells + 1))
                brush_reachable = cv2.dilate(
                    self._reachable_mask.astype(np.uint8), brush_kernel, iterations=1)
                free_space = (walls_closed == 0)
                self._coverage_mask = (brush_reachable > 0) & free_space
                self.get_logger().info(
                    f'📊 掩码已缓存 | '
                    f'安全可达(目标选择): {np.sum(self._reachable_mask)} | '
                    f'刷子可达(覆盖率分母): {np.sum(self._coverage_mask)} | '
                    f'全部自由区(旧分母): {np.sum(free_space)} | '
                    f'总像素: {self.rows * self.cols}')
            
            covered_cells_reachable = np.sum((self.grid_data == 50) & self._coverage_mask)
            total_flippable_cells = np.sum(self._coverage_mask)
            
            if total_flippable_cells == 0: 
                total_flippable_cells = 1
            
            cov_pct = (covered_cells_reachable / total_flippable_cells) * 100.0
            cov_pct = min(100.0, cov_pct)
        else:
            covered_cells = np.sum(self.grid_data == 50)
            total_cells = self.rows * self.cols
            cov_pct = (covered_cells / total_cells) * 100.0
        
        status = CoverageStatus()
        status.coverage_percentage = cov_pct
        self.status_pub.publish(status)
        self.coverage_percentage = cov_pct

    def map_callback(self, msg):
        self.res = msg.info.resolution
        self.cols = msg.info.width
        self.rows = msg.info.height
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        # 保存完整的原点位姿，包含旋转
        self.origin_pose = msg.info.origin
        self.width_m = self.cols * self.res
        self.height_m = self.rows * self.res
        # 对应地，机器人的 0.5m 物理喷洒半径也动态除以真实的分辨率
        self.work_radius_cells = int(0.65 / self.res) 

        raw_data = np.array(msg.data, dtype=np.int8).reshape(self.rows, self.cols)
        self.static_map_data = raw_data.copy()
        
        # 如果是第一次或者地图尺寸突变
        if self.grid_data is None or self.grid_data.shape != (self.rows, self.cols):
             # 初始训练网格应继承静态地图的障碍信息，避免只发布空白网格
             self.grid_data = self.static_map_data.copy()

    # 发布hud信息
    def publish_hud(self):
        if not hasattr(self, 'cv_bridge') or self.cv_bridge is None:
            return
        
        hud_img = np.zeros((180, 320, 3), dtype=np.uint8)
        hud_img[:] = (30, 20, 20)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        
        alg_name = getattr(self, 'alg_name', "HMS-CPP Algorithm")
        cv2.putText(hud_img, alg_name, (10, 30), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud_img, f"Coverage   {self.coverage_percentage:.2f} %", (10, 70), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        elapsed = time.time() - self.episode_start_time if getattr(self, 'episode_start_time', None) else 0.0
        cv2.putText(hud_img, f"Time       {elapsed:.1f} s", (10, 105), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        phase = getattr(self, 'current_phase', 'Unknown')
        cv2.putText(hud_img, f"Phase      {phase}", (10, 140), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        status = getattr(self, 'current_status', 'RUNNING')
        color = (0, 255, 255) if status == "RUNNING" else (0, 165, 255) if status == "FINISHED" else (200, 200, 200)
        cv2.putText(hud_img, f"Status     {status}", (10, 175), font, font_scale, color, thickness, cv2.LINE_AA)
        
        img_msg = self.cv_bridge.cv2_to_imgmsg(hud_img, encoding="bgr8")
        self.hud_pub.publish(img_msg)

    def reset_local_grid(self):

        if self.static_map_data is not None:
             self.grid_data = self.static_map_data.copy()
        else:
             return
        
        # 清除所有形态学掩码缓存，确保新 Episode/地图时重建
        self._obstacle_inflated = None
        self._reachable_mask = None
        self._coverage_mask = None
             
        # 四壁加厚封锁 (使用动态获取的长宽边界)
        self.grid_data[0, :] = 100
        self.grid_data[-1, :] = 100
        self.grid_data[:, 0] = 100
        self.grid_data[:, -1] = 100

    def step_sac(self):
        """巧妙利用已有的 q_values 发送属于 SAC 的 [dx, dy] 连续矢量"""
        req = SACPredict.Request()
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'map'
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.info.resolution = float(self.res)
        grid_msg.info.width = self.cols
        grid_msg.info.height = self.rows
        grid_msg.info.origin.position.x = float(self.origin_x)
        grid_msg.info.origin.position.y = float(self.origin_y)
        if hasattr(self, 'origin_pose'):
            grid_msg.info.origin.orientation = self.origin_pose.orientation
        else:
            grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = self.grid_data.flatten().tolist()
        req.occupancy_grid = grid_msg
        
        pose = self.get_current_pose()
        if pose is None:
            return None, 0.0, 0.0
        req.robot_pose = pose
        req.coverage_status.coverage_percentage = self.coverage_percentage
        
        # 【行为克隆】将上次的实际执行动作和重写标志传给 RL 服务端
        req.robot_velocity.linear.x = float(self.last_actual_fwd)
        req.robot_velocity.linear.y = float(self.last_actual_left)
        req.coverage_status.repeated_coverage_count = 1 if self.action_was_overridden else 0
        
        # 传递完后立刻重置标志
        self.action_was_overridden = False
        
        future = self.sac_client.call_async(req)
        wait_start = time.time()
        while not future.done() and (time.time() - wait_start) < 3.0: # 增加到 3 秒超时
            time.sleep(0.05)
        
        if future.done():
            if future.result() is not None:
                resp = future.result()
                if len(resp.q_values) >= 2:
                    return resp.action, resp.q_values[0], resp.q_values[1]
                else:
                    self.get_logger().error('⚠️ SAC 返回值 q_values 长度不足！')
                    return 0, 0.0, 0.0
            else:
                self.get_logger().error('⚠️ SAC 服务端返回了 None！')
                return None, 0.0, 0.0
        else:
            self.get_logger().error('🚨 SAC 服务端超时无响应！请检查 sac_server_grid 节点是否崩溃。')
            return None, 0.0, 0.0


from gazebo_msgs.srv import SetEntityState, GetEntityState

def generate_boustrophedon_waypoints(core_mask, res, origin_x, origin_y, width_m, height_m, 
                                    rows, cols, step_size, curr_pose=None, logger=None):
    """
    【独立的几何规划算法】：从安全核心生成弓字形航点
    
    参数:
      core_mask: OpenCV腐蚀后的安全区域掩码
      res: 地图分辨率
      origin_x, origin_y: 地图原点
      width_m, height_m: 地图物理尺寸
      rows, cols: 网格尺寸
      step_size: 扫描行间距(米)
      curr_pose: 当前位置 (PoseStamped 或 None)
      logger: 日志输出器
      
    返回:
      waypoints: [(x, y, yaw), ...]
    """
    waypoints = []
    step_px = int(step_size / res)
    
    # 计算安全边界 (0.60m: 避开 0.52m 的 Nav2 致命膨胀层，防止机器人贴墙过紧导致碰撞或目标点被抛弃) 控制几何规划的贴墙距离
    safe_margin = 0.6
    map_x_min = origin_x + safe_margin
    map_x_max = origin_x + width_m - safe_margin
    map_y_min = origin_y + safe_margin
    map_y_max = origin_y + height_m - safe_margin
    
    # 确定航向：寻找离机器人最近的“极点”（顶部或底部），从那里开始一气呵成的全程扫掠
    valid_y_indices = [y for y in range(rows) if np.any(core_mask[y, :] > 127)]
    if not valid_y_indices:
        return []
        
    min_valid_y = valid_y_indices[0]
    max_valid_y = valid_y_indices[-1]

    y_indices = []
    if curr_pose is not None:
        curr_y_idx = int((curr_pose.pose.position.y - origin_y) / res)
        
        # 判断离南边近还是北边近
        if curr_y_idx > (rows / 2):
            # 离北边近，从最北行的有效边界往下扫
            y_indices = list(range(max_valid_y, min_valid_y - 1, -step_px))
            if y_indices[-1] != min_valid_y:
                y_indices.append(min_valid_y)
            if logger:
                logger.info(f'🔄 确定扫掠航向：从北部实际安全边界开始，向南扫描。')
        else:
            # 离南边近，从最南行的有效边界往上扫
            y_indices = list(range(min_valid_y, max_valid_y + 1, step_px))
            if y_indices[-1] != max_valid_y:
                y_indices.append(max_valid_y)
            if logger:
                logger.info(f'🔄 确定扫掠航向：从南部实际安全边界开始，向北扫描。')
    else:
        # 默认从底边界往上
        y_indices = list(range(min_valid_y, max_valid_y + 1, step_px))
        if y_indices[-1] != max_valid_y:
            y_indices.append(max_valid_y)

    # 智能判断初始横向扫掠方向 (是从左往右，还是从右往左)
    # 默认从左到右
    direction = 1 
    
    # 如果已知当前位置，看看是在田块偏左还是偏右，那就从离它近的那头开始扫
    if curr_pose is not None:
        curr_x = curr_pose.pose.position.x
        mid_x = (map_x_min + map_x_max) / 2.0
        if curr_x > mid_x:
            direction = -1 # 偏右，首行从右往左扫
            if logger: logger.info("➡️ 判断首行动向：偏右侧，首行将 [自东向西] 扫掠")
        else:
            if logger: logger.info("⬅️ 判断首行动向：偏左侧，首行将 [自西向东] 扫掠")

    for y_idx in y_indices:
        row = core_mask[y_idx, :]
        free_ids = np.where(row > 127)[0]
        if len(free_ids) == 0:
            continue
        
        raw_min_x = origin_x + free_ids[0] * res
        raw_max_x = origin_x + free_ids[-1] * res
        y_pos = origin_y + y_idx * res
        
        min_x = np.clip(raw_min_x, map_x_min, map_x_max)
        max_x = np.clip(raw_max_x, map_x_min, map_x_max)
        y_pos = np.clip(y_pos, map_y_min, map_y_max)

        if max_x - min_x < 1.0: 
            continue
        
        # direction = 1: 从左扫到右
        # direction = -1: 从右扫到左
        if direction == 1:
            waypoints.append((min_x, y_pos, 0.0))
            waypoints.append((max_x, y_pos, 0.0))
        else:
            waypoints.append((max_x, y_pos, 3.14))
            waypoints.append((min_x, y_pos, 3.14))
            
        # 每一行扫完，方向必须彻底反转！这样才能画出蛇形弓字
        direction *= -1

    # 计算每个航点的yaw指向下一个航点
    for i in range(len(waypoints) - 1):
        curr_wp = waypoints[i]
        next_wp = waypoints[i + 1]
        dx = next_wp[0] - curr_wp[0]
        dy = next_wp[1] - curr_wp[1]
        yaw = math.atan2(dy, dx)
        waypoints[i] = (curr_wp[0], curr_wp[1], yaw)
    
    if len(waypoints) > 1:
        last_wp = waypoints[-1]
        prev_wp = waypoints[-2]
        dx = last_wp[0] - prev_wp[0]
        dy = last_wp[1] - prev_wp[1]
        yaw = math.atan2(dy, dx)
        waypoints[-1] = (last_wp[0], last_wp[1], yaw)

    return waypoints


def reset_environment(node, nav, is_first_init=False):
    node.get_logger().info('🔄 RESETTING ENVIRONMENT...')
    if not is_first_init:
        nav.cancelTask()
    node.reset_local_grid()
    node.reset_trajectory()
    
    for srv_name, srv_type in [('/reset_sac_state', Empty), ('/reset_coverage', Empty)]:
        cli = node.create_client(srv_type, srv_name)
        if cli.wait_for_service(timeout_sec=1.0):
            future = cli.call_async(srv_type.Request())
            wait_start = time.time()
            while not future.done() and (time.time() - wait_start) < 1.0:
                time.sleep(0.02)

    gazebo_set_cli = node.create_client(SetEntityState, '/set_entity_state')
    gazebo_get_cli = node.create_client(GetEntityState, '/get_entity_state')
    
    if getattr(node, 'home_pose_x', None) is None:
        node.get_logger().info('📥 第一次启动，正在从 Gazebo 绝对坐标，打破 AMCL TF 死锁...')
        if gazebo_get_cli.wait_for_service(timeout_sec=3.0):
            req_get = GetEntityState.Request()
            req_get.name = 'fishbot'
            req_get.reference_frame = 'world'
            future_get = gazebo_get_cli.call_async(req_get)
            wait_start = time.time()
            while not future_get.done() and (time.time() - wait_start) < 2.0:
                time.sleep(0.02)
            if future_get.done() and future_get.result().success:
                node.home_pose_x = future_get.result().state.pose.position.x
                node.home_pose_y = future_get.result().state.pose.position.y
                # 简单近似yaw，为了保险我们默认为0.0或者直接从Quaternion解算
                node.home_pose_yaw = 0.0 
                node.get_logger().info(f'🎯 成功锁定 Gazebo 物理坐标: ({node.home_pose_x:.2f}, {node.home_pose_y:.2f})')
            else:
                node.get_logger().error('❌ 无法从 Gazebo 获取坐标，回落到远古 0.0 原点。')
                node.home_pose_x, node.home_pose_y, node.home_pose_yaw = 0.0, 0.0, 0.0

    target_x = getattr(node, 'home_pose_x', 0.0)
    target_y = getattr(node, 'home_pose_y', 0.0)
    target_yaw = getattr(node, 'home_pose_yaw', 0.0)
    
    # 瞬移机器人到大本营
    if gazebo_set_cli.wait_for_service(timeout_sec=1.0):
        req = SetEntityState.Request()
        req.state.name = 'fishbot'
        req.state.pose.position.x = float(target_x)
        req.state.pose.position.y = float(target_y)
        req.state.pose.orientation.z = math.sin(target_yaw / 2.0)
        req.state.pose.orientation.w = math.cos(target_yaw / 2.0)
        req.state.reference_frame = 'world'
        future = gazebo_set_cli.call_async(req)
        wait_start = time.time()
        while not future.done() and (time.time() - wait_start) < 1.0:
            time.sleep(0.02)

    time.sleep(1.0)
    time.sleep(0.5) 
    
    current_pose = node.get_current_pose()
    if current_pose is not None:
        dist = math.sqrt((current_pose.pose.position.x - target_x)**2 + 
                         (current_pose.pose.position.y - target_y)**2)
        node.get_logger().info(f'✅ P3D 真值校准瞬间完成！误差: {dist:.5f}m')
    else:
        node.get_logger().warn(f'⚠️ TF 未就绪')
    
    # 初始化覆盖图并发布
    current_pose = node.get_current_pose()
    if current_pose is not None:
        node.update_coverage(current_pose.pose.position.x, current_pose.pose.position.y)
    node.publish_grid()


def main():
    rclpy.init()
    trainer = SACTrainerClient()
    nav = BasicNavigator()

    executor = MultiThreadedExecutor()
    executor.add_node(trainer)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    time.sleep(1.0) 
    episode = 0
    
    try:
        trainer.get_logger().info('🚀 Hybrid Architecture ONLINE: Mathematical Core + ML Edge')
        trainer.get_logger().info('⬆️ 正在等待 Nav2 核心控制器与 P3D 真值矩阵上线...')
        time.sleep(5.0)
        trainer.get_logger().info('✅ Nav2 纯净导航打通！P3D 上帝视角 TF 树已建立！')
        
        while rclpy.ok():
            episode_start_time = time.time()
            trainer.episode_start_time = episode_start_time
            trainer.current_status = 'RUNNING'
            trainer.current_phase = 'Initializing...'
            trainer.get_logger().info(f'🎬 --- Episode {episode} Start ---')
            time.sleep(1.0)
            
            while trainer.static_map_data is None:
                trainer.get_logger().info('等待地图加载...')
                time.sleep(1.0)


            # 全局地图动态腐蚀提取法
            free_mask = np.zeros_like(trainer.static_map_data, dtype=np.uint8)
            # 只认 0（确认自由空间），排除 unknown(-1/205) 和占用(100) 区域
            free_mask[trainer.static_map_data == 0] = 255
            
            # 安全裕度：退缩内缩 0.6 米 (与安全裁剪 safe_margin 保持一致，避开致命膨胀层)
            erosion_px = int(0.6 / trainer.res) 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2, erosion_px * 2))
            core_mask = cv2.erode(free_mask, kernel, iterations=1)
            
            # ====== Phase 1: 几何弓字形铺路 ======
            # 计算地图的安全物理边界（内缩 0.55m 以极限贴近膨胀层）
            safe_margin = 0.6
            map_x_min = trainer.origin_x + safe_margin
            map_x_max = trainer.origin_x + trainer.width_m - safe_margin
            map_y_min = trainer.origin_y + safe_margin
            map_y_max = trainer.origin_y + trainer.height_m - safe_margin
            
            trainer.current_phase = 'Geometric'
            trainer.get_logger().info(
                f'🗺️ Phase 1: 弓字形几何规划器\\n'
                f'   地图原点: ({trainer.origin_x:.2f}, {trainer.origin_y:.2f})\\n'
                f'   地图物理尺寸: {trainer.width_m:.1f}m x {trainer.height_m:.1f}m\\n'
                f'   安全航行区域: X[{map_x_min:.2f}, {map_x_max:.2f}] Y[{map_y_min:.2f}, {map_y_max:.2f}]'
            )
            
            # 【调用独立的几何规划函数】
            curr_pose = trainer.get_current_pose()
            waypoints = generate_boustrophedon_waypoints(
                core_mask, 
                trainer.res, 
                trainer.origin_x, 
                trainer.origin_y,
                trainer.width_m, 
                trainer.height_m,
                trainer.rows, 
                trainer.cols,
                trainer.step_size,
                curr_pose,
                trainer.get_logger()
            )

            # 【支持直接训练 Phase 2】
            skip_phase1 = trainer.get_parameter('skip_phase1').get_parameter_value().bool_value
            if skip_phase1:
                waypoints = []
                trainer.current_phase = 'SAC Recovery'
                trainer.get_logger().info("⏭️ 参数 skip_phase1=True，跳过几何铺路，直接启动 SAC 全图探索！")

            trainer.get_logger().info(f'⚙️ 共 {len(waypoints)} 个航点。执行 Phase 1...')
            
            # 为第一个航点计算正确的yaw，使其直接指向目标
            if len(waypoints) > 0 and curr_pose is not None:
                first_wp = waypoints[0]
                dx = first_wp[0] - curr_pose.pose.position.x
                dy = first_wp[1] - curr_pose.pose.position.y
                first_yaw = math.atan2(dy, dx)
                waypoints[0] = (first_wp[0], first_wp[1], first_yaw)
                trainer.get_logger().info(f'🎯 第一航点yaw已自动校准: {math.degrees(first_yaw):.1f}°')
            
            nav_aborted = False
            for wp_idx, wp in enumerate(waypoints):
                if not rclpy.ok():
                    break
                
                # 记录航点开始时的覆盖率
                coverage_at_start = trainer.coverage_percentage
                trainer.get_logger().info(f'📍 航点 {wp_idx}/{len(waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f}) | 起始覆盖率: {coverage_at_start:.1f}%')
                
                goal = trainer.xyy_to_pose(wp[0], wp[1], wp[2])
                nav.goToPose(goal)
                
                while not nav.isTaskComplete():
                    if not rclpy.ok(): break
                    time.sleep(0.1)
                
                # 航点完成后打印覆盖率增长
                coverage_at_end = trainer.coverage_percentage
                coverage_gain = coverage_at_end - coverage_at_start
                trainer.get_logger().info(f'✅ 航点 {wp_idx} 完成 | 覆盖率: {coverage_at_start:.1f}% → {coverage_at_end:.1f}% (+{coverage_gain:.1f}%)')
                
                # 【新增】：将 Phase 1 每步的覆盖率直接追加到 progress.csv 中
                if hasattr(trainer, 'progress_csv_path') and trainer.progress_csv_path:
                    try:
                        with open(trainer.progress_csv_path, 'a') as f:
                            f.write(f"0.0,{coverage_at_end}\n")
                            f.flush()
                    except Exception as e:
                        pass

            # 检查是否为纯几何测试模式，如果是则跳过 Phase 2
            stop_after_phase1 = trainer.get_parameter('stop_after_phase1').get_parameter_value().bool_value
            if stop_after_phase1:
                trainer.get_logger().info(f'🏁 纯几何测试模式 (stop_after_phase1=True)，Phase 1 已完成，直接进入结算环节！')
            else:
                trainer.current_phase = 'SAC Recovery'
                trainer.get_logger().info(f'🤖 Phase 2 [Hand-off]: 数学骨架耗净，激活最高级 SAC 不规则边缘特种侦察兵！')
                
            stuck_count = 0
            last_coverage_check = trainer.coverage_percentage
            
            while rclpy.ok() and trainer.coverage_percentage < trainer.target_coverage and not stop_after_phase1:
                action_idx, dx, dy = trainer.step_sac()
                if dx is None:
                    continue
                    
                # 如果机器人彻底卡死（Nav2判定目标不可达瞬间退出，且覆盖率毫无寸进）
                # 必须在尝试 5 次无效动作后强制掐断 Episode，防止死循环无限输出日志
                if trainer.coverage_percentage - last_coverage_check < 0.01:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    last_coverage_check = trainer.coverage_percentage
                    
                if stuck_count > 5:
                    trainer.get_logger().error(f'🚨 检测到 ({stuck_count}次行动徒劳无功)！强制终结本回合！')
                    nav.cancelTask()
                    break
                    
                curr_pose = trainer.get_current_pose()
                if not curr_pose:
                    continue
                    
                cx = curr_pose.pose.position.x
                cy = curr_pose.pose.position.y
                # 获取当前朝向 (Yaw)
                q = curr_pose.pose.orientation
                cyaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
                
                # ===== 检查地图参数是否已初始化 =====
                if trainer.width_m is None or trainer.height_m is None:
                    trainer.get_logger().warn('⚠️ 地图参数未初始化，跳过本次SAC动作')
                    continue
                
                # SAC 完全控制目标点选择
                # SAC 输出 (fwd, left) ∈ [-1, 1]²，直接映射为 body frame 位移
                # 幅度 = 距离，方向 = 朝向，不再做归一化截断
                fwd = dx   # action[0]: 前进/后退
                left = dy  # action[1]: 左/右
                
                # 地图尺度：SAC 输出 1.0 对应地图最大物理长度 (确保能从一角跨到另一角)
                map_scale = max(trainer.width_m, trainer.height_m) * 1.0
                
                # 直接用 SAC 原始输出 × 地图尺度，不归一化
                # 这样 SAC 同时控制方向和距离：
                #   输出 [0.1, 0] → 短距离前进（清理近处碎片）
                #   输出 [1.0, 0] → 长直线穿越（覆盖大片区域）
                #   输出 [0.5, 0.5] → 中距离斜向移动
                scaled_fwd = fwd * map_scale
                scaled_left = left * map_scale
                
                # 使用当前 Yaw 角进行旋转变换 (body → world)
                world_dx = scaled_fwd * math.cos(cyaw) - scaled_left * math.sin(cyaw)
                world_dy = scaled_fwd * math.sin(cyaw) + scaled_left * math.cos(cyaw)

                tx = cx + world_dx
                ty = cy + world_dy
                # 航点朝向设置为位移方向
                target_yaw = math.atan2(world_dy, world_dx)
                
                # 边距必须 >= Nav2 costmap inflation_radius (0.55m)
                # 否则 Nav2 在规划边缘目标时，膨胀层会访问地图外的像素导致 worldToMap failed
                # 注意：这个裁剪只管"别太靠近地图边框"，实际形状适配由下方障碍物掩码负责
                safety_margin = 0.5
                min_x_limit = trainer.origin_x + safety_margin
                max_x_limit = trainer.origin_x + trainer.width_m - safety_margin
                min_y_limit = trainer.origin_y + safety_margin
                max_y_limit = trainer.origin_y + trainer.height_m - safety_margin
                
                # 确保裁剪后的点确实在地图内
                clipped_tx = np.clip(tx, min_x_limit, max_x_limit)
                clipped_ty = np.clip(ty, min_y_limit, max_y_limit)
                
                expert_override = False

                # 静态地图障碍物安全过滤
                # 矩形边界只卡外框，这一步检查目标是否落在墙壁/柱子的膨胀区内
                if trainer.static_map_data is not None:
                    # 缓存膨胀掩码，避免每步重复计算
                    if not hasattr(trainer, '_obstacle_inflated') or trainer._obstacle_inflated is None:
                        walls = np.logical_or(
                            trainer.static_map_data == 100, 
                            trainer.static_map_data == -1
                        ).astype(np.uint8)
                        # 【形态学闭运算】先膨胀后腐蚀（closing）
                        # 目的：连接靠近的墙壁碎片、填补障碍物之间的小缝隙
                        # 防止机器人试图穿越物理上不可通行的狭窄通道
                        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, close_kernel)
                        # 最终安全膨胀（iterations=12 ≈ 0.6m），为 0.5m 的机身增加 0.1m 安全余量
                        trainer._obstacle_inflated = cv2.dilate(walls_closed, np.ones((3,3), np.uint8), iterations=12)
                        trainer.get_logger().info('形态学闭运算完成')
                    obstacle_inflated = trainer._obstacle_inflated
                    
                    if (clipped_tx >= min_x_limit and clipped_tx <= max_x_limit and
                        clipped_ty >= min_y_limit and clipped_ty <= max_y_limit):
                        tgx = int((clipped_tx - trainer.origin_x) / trainer.res)
                        tgy = int((clipped_ty - trainer.origin_y) / trainer.res)
                        
                        # 防止栅格坐标越界
                        tgx = np.clip(tgx, 0, trainer.cols - 1)
                        tgy = np.clip(tgy, 0, trainer.rows - 1)
                        
                        # 如果目标落在障碍物膨胀区内，沿相同方向缩短步长找安全点
                        if (0 <= tgx < trainer.cols and 0 <= tgy < trainer.rows 
                            and obstacle_inflated[tgy, tgx] > 0):
                            
                            actual_dist = math.hypot(world_dx, world_dy)
                            found_safe = False
                            safe_tx, safe_ty = cx, cy
                            
                            # 【核心重构】：改用 0.2m 物理步长的射线探测，取代粗糙的百分比截断
                            # 从起点出发，顺着 SAC 指定的方向一步步向外摸索，直到撞墙
                            step_m = 0.2
                            for d in np.arange(step_m, actual_dist, step_m):
                                ratio = d / actual_dist
                                stx = cx + world_dx * ratio
                                sty = cy + world_dy * ratio
                                
                                stx = np.clip(stx, min_x_limit, max_x_limit)
                                sty = np.clip(sty, min_y_limit, max_y_limit)
                                sgx = int((stx - trainer.origin_x) / trainer.res)
                                sgy = int((sty - trainer.origin_y) / trainer.res)
                                sgx = np.clip(sgx, 0, trainer.cols - 1)
                                sgy = np.clip(sgy, 0, trainer.rows - 1)
                                
                                if obstacle_inflated[sgy, sgx] > 0:
                                    break # 探测到障碍物，停止前进，保留上一个探明的安全点
                                else:
                                    safe_tx, safe_ty = stx, sty
                                    found_safe = True
                            
                            if not found_safe:
                                escape_yaw = cyaw + np.random.uniform(math.pi/2, 3*math.pi/2)
                                clipped_tx = np.clip(cx + 0.8 * math.cos(escape_yaw), min_x_limit, max_x_limit)
                                clipped_ty = np.clip(cy + 0.8 * math.sin(escape_yaw), min_y_limit, max_y_limit)
                                expert_override = True
                            else:
                                clipped_tx, clipped_ty = safe_tx, safe_ty
                                safe_dist = math.hypot(safe_tx - cx, safe_ty - cy)
                                expert_override = True

                target_grid_x = int((clipped_tx - trainer.origin_x) / trainer.res)
                target_grid_y = int((clipped_ty - trainer.origin_y) / trainer.res)
                # 确保栅格坐标不会越界
                target_grid_x = np.clip(target_grid_x, 0, trainer.cols - 1)
                target_grid_y = np.clip(target_grid_y, 0, trainer.rows - 1)
                target_in_uncovered = False
                target_cell = -1
                
                if (0 <= target_grid_x < trainer.cols and 
                    0 <= target_grid_y < trainer.rows):
                    target_cell = trainer.grid_data[target_grid_y, target_grid_x]
                    # 检查目标是否在未覆盖区 (0) 或边缘区 (未被识别为障碍的区域)
                    if target_cell == 0 or (target_cell >= 1 and target_cell < 50):
                        target_in_uncovered = True
                
                # 如果是纯启发式模式，强制屏蔽 SAC 的动作
                use_heuristic = trainer.get_parameter('use_heuristic').get_parameter_value().bool_value
                if use_heuristic:
                    target_in_uncovered = False
                
                if not target_in_uncovered:
                    expert_override = True
                    try:
                        # 1. 已覆盖区域掩码
                        covered = (trainer.grid_data == 50)
                        
                        # 2. 模拟膨胀：将 covered 矩阵向上下左右平移一格并取并集
                        dilated_cov = np.zeros_like(covered)
                        dilated_cov[1:, :] |= covered[:-1, :] # 下移
                        dilated_cov[:-1, :] |= covered[1:, :] # 上移
                        dilated_cov[:, 1:] |= covered[:, :-1] # 右移
                        dilated_cov[:, :-1] |= covered[:, 1:] # 左移
                        dilated_cov |= covered
                        
                        # 3. 前沿 = 膨胀后的覆盖区 与 真实未覆盖区 的交集
                        frontier_mask = dilated_cov & (trainer.grid_data == 0)
                        
                        # 4. 排除静态障碍物
                        obstacle_inflated = getattr(trainer, '_obstacle_inflated', None)
                        if obstacle_inflated is not None:
                            frontier_mask = frontier_mask & (obstacle_inflated == 0)
                            
                        # 5. 贪心搜索最近前沿点
                        fy, fx = np.where(frontier_mask)
                        if len(fx) > 0:
                            cgx = int((cx - trainer.origin_x) / trainer.res)
                            cgy = int((cy - trainer.origin_y) / trainer.res)
                            
                            # 计算当前机器人位置到所有前沿点的距离平方
                            dists = (fx - cgx)**2 + (fy - cgy)**2
                            best_idx = np.argmin(dists)
                            best_gx = fx[best_idx]
                            best_gy = fy[best_idx]
                            
                            clipped_tx = best_gx * trainer.res + trainer.origin_x
                            clipped_ty = best_gy * trainer.res + trainer.origin_y
                            target_yaw = math.atan2(clipped_ty - cy, clipped_tx - cx)
                        else:
                            if use_heuristic:
                                trainer.get_logger().info('🏁 [纯启发式] 全图已无可达前沿点，探测结束！')
                                nav.cancelTask()
                                break
                    except Exception as e:
                        trainer.get_logger().error(f'前沿搜索失败: {e}')
                
                # --- 【行为克隆】计算实际执行的动作 ---
                # 只有真正触发了障碍物避让或前沿拉回时，才标记为专家接管
                if expert_override:
                    trainer.action_was_overridden = True
                    actual_world_dx = clipped_tx - cx
                    actual_world_dy = clipped_ty - cy
                    
                    actual_scaled_fwd = actual_world_dx * math.cos(cyaw) + actual_world_dy * math.sin(cyaw)
                    actual_scaled_left = -actual_world_dx * math.sin(cyaw) + actual_world_dy * math.cos(cyaw)
                    
                    trainer.last_actual_fwd = np.clip(actual_scaled_fwd / map_scale, -1.0, 1.0)
                    trainer.last_actual_left = np.clip(actual_scaled_left / map_scale, -1.0, 1.0)
                else:
                    trainer.action_was_overridden = False
                    trainer.last_actual_fwd = fwd
                    trainer.last_actual_left = left
                # --------------------------------------
                
                goal = trainer.xyy_to_pose(clipped_tx, clipped_ty, target_yaw)
                nav.goToPose(goal)
                
                while not nav.isTaskComplete():
                    time.sleep(0.05)
                    if trainer.coverage_percentage >= trainer.target_coverage:
                        nav.cancelTask()
                        break
                        
                if trainer.coverage_percentage >= trainer.target_coverage:
                        nav.cancelTask()
                        break
                
            trainer.get_logger().info(f'✅ Episode {episode} 最终落下帷幕. Coverage: {trainer.coverage_percentage:.1f}%')
            
            # --- Calculate Evaluation Metrics ---
            total_time_s = time.time() - episode_start_time
            path_length = trainer.total_path_length
            
            # 重复覆盖率：基于路径效率公式
            # R_r = (L_actual - L_ideal) / L_ideal × 100%
            # L_ideal = A_covered / W_step（以行间距为有效宽度的理论最短路径）
            actual_covered_area = np.sum(trainer.grid_data == 50) * (trainer.res ** 2)
            ideal_path_length = actual_covered_area / trainer.step_size if trainer.step_size > 0 else path_length
            if ideal_path_length > 0:
                repeated_coverage = max(0.0, (path_length / ideal_path_length - 1)) * 100.0
            else:
                repeated_coverage = 0.0
                
            trainer.get_logger().info('='*50)
            trainer.get_logger().info('📝 [Evaluation Metrics]')
            trainer.get_logger().info(f'Global Coverage Ratio (C_r)   : {trainer.coverage_percentage:.2f}%')
            trainer.get_logger().info(f'Path Length (L)               : {path_length:.2f} meters')
            trainer.get_logger().info(f'Repeated Coverage Ratio (R_r) : {repeated_coverage:.2f}%')
            trainer.get_logger().info(f'Total Time (T)                : {total_time_s:.2f} seconds')
            trainer.get_logger().info('='*50)
            
            # --- Save Metrics to CSV ---
            if hasattr(trainer, 'progress_csv_path') and trainer.progress_csv_path:
                metrics_csv_path = os.path.join(os.path.dirname(trainer.progress_csv_path), 'evaluation_metrics.csv')
                try:
                    with open(metrics_csv_path, 'w') as f:
                        f.write("Global_Coverage_Ratio,Path_Length_m,Repeated_Coverage_Ratio,Total_Time_s\n")
                        f.write(f"{trainer.coverage_percentage:.2f},{path_length:.2f},{repeated_coverage:.2f},{total_time_s:.2f}\n")
                    trainer.get_logger().info(f'📊 评估指标已保存至: {metrics_csv_path}')
                except Exception as e:
                    trainer.get_logger().error(f'保存评估指标失败: {e}')
            
            
            if trainer.mode == 'train':
                trainer.get_logger().info('🔄 训练完成，退出当前节点交还控制权给课程大师...')
                break
            else:
                trainer.get_logger().info('🏁 推理目标达成！机器人已停止，窗口保持打开供截图。按 Ctrl+C 退出。')
                # 推理模式下不退出节点，保持 Gazebo/RViz 打开供用户截图
                while rclpy.ok():
                    time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
