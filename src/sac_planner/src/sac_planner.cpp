#include "sac_planner/sac_planner.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <functional>
#include <tf2/utils.h>

PLUGINLIB_EXPORT_CLASS(sac_planner::SACPlanner, nav2_core::GlobalPlanner)

namespace sac_planner
{

SACPlanner::SACPlanner()
: node_(nullptr),
  tf_(nullptr),
  costmap_ros_(nullptr),
  costmap_(nullptr),
  tolerance_(0.5),
  use_astar_(false),
  allow_unknown_(true),
  work_width_(1.0)
{
}

SACPlanner::~SACPlanner()
{
}

void SACPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Configuring SACPlanner");

  node_ = parent.lock();
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();

  // 获取配置参数
  node_->declare_parameter(name_ + ".tolerance", 0.5);
  node_->declare_parameter(name_ + ".use_astar", false);
  node_->declare_parameter(name_ + ".allow_unknown", true);
  node_->declare_parameter(name_ + ".work_width", 1.0);

  node_->get_parameter(name_ + ".tolerance", tolerance_);
  node_->get_parameter(name_ + ".use_astar", use_astar_);
  node_->get_parameter(name_ + ".allow_unknown", allow_unknown_);
  node_->get_parameter(name_ + ".work_width", work_width_);

  // 初始化覆盖地图发布者
  coverage_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>("covered_grid", 10);
  passed_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("passed_path", 10);
  coverage_status_pub_ = node_->create_publisher<agricultural_msgs::msg::CoverageStatus>("coverage_status", 10);

  // 初始化重置覆盖率服务
  reset_service_ = node_->create_service<std_srvs::srv::Empty>(
    "reset_coverage",
    std::bind(&SACPlanner::resetCoverageCallback, this, std::placeholders::_1, std::placeholders::_2));

  // 初始化覆盖地图数据
  initCoverageGrid();

  // 注意：移除了定时器，改为在createPlan()中同步更新覆盖状态
  // 原因：定时器在节点未完全激活时运行会导致时钟未初始化错误
  // "rcl_wait() failed: Clock is not initialized"

  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "SACPlanner configured successfully");
}

void SACPlanner::cleanup()
{
  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Cleaning up SACPlanner");
  sac_predict_client_.reset();
}

void SACPlanner::activate()
{
  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Activating SACPlanner");
  initializeDqnClient();
}

void SACPlanner::deactivate()
{
  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Deactivating SACPlanner");
  sac_predict_client_.reset();
}

nav_msgs::msg::Path SACPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  // 1. 同步更新覆盖图（确保数据最新）
  geometry_msgs::msg::PoseStamped current_pose;
  rclcpp::Time now = node_->get_clock()->now(); // 强制使用节点时钟（确保 sim_time 生效）

  try {
    geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
      costmap_ros_->getGlobalFrameID(), 
      costmap_ros_->getBaseFrameID(),
      tf2::TimePointZero);
    
    current_pose.header.frame_id = costmap_ros_->getGlobalFrameID();
    current_pose.header.stamp = transform.header.stamp;
    current_pose.pose.position.x = transform.transform.translation.x;
    current_pose.pose.position.y = transform.transform.translation.y;
    current_pose.pose.position.z = transform.transform.translation.z;
    current_pose.pose.orientation = transform.transform.rotation;
    
    updateCoverageGrid(current_pose);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "Could not transform base_link to map: %s", ex.what());
    current_pose = start;
    current_pose.header.stamp = now;
  }

  // 2. 发送服务请求到 SAC Python Bridge
  RCLCPP_INFO(node_->get_logger(), "Creating plan from SAC at time %.2f", now.seconds());
  
  nav_msgs::msg::Path path;
  path.header.stamp = now;
  path.header.frame_id = costmap_ros_->getGlobalFrameID();

  // 检查成本地图是否有效
  if (!costmap_ || !node_) {
    RCLCPP_ERROR(node_->get_logger(), "Costmap or node not initialized");
    return path;
    }

  // 检查SAC服务客户端是否有效 
  if (!sac_predict_client_ || !sac_predict_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_WARN(node_->get_logger(), "SAC service not available, using fallback plan");
    path.poses.push_back(start);
    path.poses.push_back(goal);
    return path;
  }

  // 获取SAC预测（传入准确的当前坐标）
  auto prediction = getSacPrediction(current_pose, goal);
  if (!prediction) {
    RCLCPP_WARN(node_->get_logger(), "SAC prediction failed, using fallback plan");
    path.poses.push_back(start);
    path.poses.push_back(goal);
    return path;
  }

  // 从SAC预测生成路径 (从当前位置开始生成)
  path = generatePathFromSacActions(current_pose, goal, prediction);
  
  // 终极修复：确保路径中的所有位姿都有相同的时间戳
  rclcpp::Time path_time = node_->get_clock()->now();
  path.header.stamp = path_time;
  for (auto & p : path.poses) {
    p.header.stamp = path_time;
  }

  RCLCPP_INFO(node_->get_logger(), "Plan created with %zu waypoints at time %.2f", path.poses.size(), path_time.seconds());
  return path;
}

void SACPlanner::initializeDqnClient()
{
  if (node_) {
    sac_predict_client_ = node_->create_client<agricultural_msgs::srv::SACPredict>(
      "sac_predict_service");
    RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "SAC predict client initialized");
  }
}

agricultural_msgs::srv::SACPredict::Response::SharedPtr SACPlanner::getSacPrediction(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  if (!sac_predict_client_) {
    RCLCPP_ERROR(rclcpp::get_logger("SACPlanner"), "SAC client not initialized");
    return nullptr;
  }

  // 创建服务请求
  auto request = std::make_shared<agricultural_msgs::srv::SACPredict::Request>();

  // 4. 任务目标
  request->robot_pose = start;
  request->goal_pose = goal;
  request->task_type = 0;  // 0: 全覆盖

  // 5. 填入当前的覆盖状态图和地图（使用互斥锁保护）
  nav_msgs::msg::OccupancyGrid combined_grid;
  {
    std::lock_guard<std::mutex> lock(coverage_mutex_);
    
    // 自动适配代价地图尺寸变化 (核心修复：防止出现 size mismatch)
    unsigned int costmap_size_x = costmap_->getSizeInCellsX();
    unsigned int costmap_size_y = costmap_->getSizeInCellsY();
    
    if (coverage_grid_.info.width != costmap_size_x || coverage_grid_.info.height != costmap_size_y) {
      RCLCPP_INFO(node_->get_logger(), "Map size changed from %dx%d to %dx%d. Syncing coverage grid.", 
                  coverage_grid_.info.width, coverage_grid_.info.height, costmap_size_x, costmap_size_y);
      initCoverageGrid(); 
    }
    
    combined_grid = coverage_grid_;
  }
  
  size_t expected_size = static_cast<size_t>(combined_grid.info.width) * combined_grid.info.height;
  const unsigned char* char_map = costmap_->getCharMap();
  for (unsigned int i = 0; i < expected_size; ++i) {
    if (char_map[i] >= nav2_costmap_2d::LETHAL_OBSTACLE) {
      combined_grid.data[i] = 100; // 障碍物标记为 100
    }
  }
  request->occupancy_grid = combined_grid;

  // 6. 填入覆盖状态（使用缓存值，避免数据竞争）
  request->coverage_status = latest_coverage_status_;

  // 发送请求并等待响应 (增加到15秒)
  auto future = sac_predict_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
    RCLCPP_ERROR(node_->get_logger(), "SAC service call timed out");
    return nullptr;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(rclcpp::get_logger("SACPlanner"), "SAC prediction failed");
    return nullptr;
  }

  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "SAC prediction received: action=%d, confidence=%.2f",
              response->action, response->confidence);
  return response;
}

nav_msgs::msg::Path SACPlanner::generatePathFromSacActions(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  const agricultural_msgs::srv::SACPredict::Response::SharedPtr & prediction)
{
  nav_msgs::msg::Path path;
  rclcpp::Time now = node_->now();
  path.header.stamp = now;
  path.header.frame_id = start.header.frame_id;

  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Path from SAC actions with timestamp: %d.%09d, use_sim_time: %d", 
              path.header.stamp.sec, path.header.stamp.nanosec, node_->get_parameter("use_sim_time").as_bool());

  // 添加起始点
  geometry_msgs::msg::PoseStamped current_step = start;
  current_step.header.stamp = now;
  path.poses.push_back(current_step);

  // ========================================
  // 对齐 SAC 连续动作空间逻辑
  // ========================================
  double dx = 0.0, dy = 0.0;
  double target_yaw = 0.0;
  const char* action_name = "CONTINUOUS_SAC";

  // 检查是否包含 SAC 输出的连续位移向量 [dx, dy]
  if (prediction->q_values.size() >= 2) {
    dx = prediction->q_values[0];
    dy = prediction->q_values[1];
    
    // 计算目标朝向 (相对于当前位姿的局部增量)
    // 注意：SAC 输出的是相对于机器人当前坐标系的位移
    target_yaw = std::atan2(dy, dx); 
    
    RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "SAC continuous action: dx=%.3f, dy=%.3f, yaw=%.3f", 
                dx, dy, target_yaw);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("SACPlanner"), "Invalid SAC prediction: expected 2 continuous values (dx, dy).");
    return path; // 返回空路径，触发重试或恢复行为
  }

  // 计算步进参数：将 SAC 的位移分解为多个局部点，方便 DWB/RPP 追踪

  int num_waypoints = 5; 
  double step_x = dx / num_waypoints;
  double step_y = dy / num_waypoints;

  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Generating path segment: action=%s, steps=%d", 
              action_name, num_waypoints);

  // 生成路径点
  for (int i = 1; i <= num_waypoints; ++i) {
    current_step.pose.position.x += step_x;
    current_step.pose.position.y += step_y;
    
    // 设置朝向
    tf2::Quaternion q;
    q.setRPY(0, 0, target_yaw);
    current_step.pose.orientation = tf2::toMsg(q);
    
    // 安全校验：碰撞检查
    unsigned int mx, my;
    if (costmap_->worldToMap(current_step.pose.position.x, current_step.pose.position.y, mx, my)) {
      if (costmap_->getCost(mx, my) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        RCLCPP_WARN(rclcpp::get_logger("SACPlanner"), "Safety Layer: Path intersects obstacle at (%.2f, %.2f)! Truncating.",
                    current_step.pose.position.x, current_step.pose.position.y);
        break;
      }
    } else {
      RCLCPP_WARN(rclcpp::get_logger("SACPlanner"), "Path point (%.2f, %.2f) is outside costmap! Truncating.",
                  current_step.pose.position.x, current_step.pose.position.y);
      break;
    }

    path.poses.push_back(current_step);
  }

  // 如果路径太短，添加一个回退点
  double dist_to_final = std::sqrt(std::pow(current_step.pose.position.x - goal.pose.position.x, 2) +
                                  std::pow(current_step.pose.position.y - goal.pose.position.y, 2));

  if (path.poses.size() <= 1 && dist_to_final > 0.5) {
    geometry_msgs::msg::PoseStamped fallback_goal = goal;
    fallback_goal.header.stamp = now;
    path.poses.push_back(fallback_goal);
  }

  RCLCPP_INFO(node_->get_logger(), "Generated path with %zu waypoints for action %s", path.poses.size(), action_name);
  return path;
}

void SACPlanner::initCoverageGrid()
{
  if (!costmap_) return;

  coverage_grid_.header.frame_id = costmap_ros_->getGlobalFrameID();
  coverage_grid_.info.resolution = costmap_->getResolution();
  coverage_grid_.info.width = costmap_->getSizeInCellsX();
  coverage_grid_.info.height = costmap_->getSizeInCellsY();
  coverage_grid_.info.origin.position.x = costmap_->getOriginX();
  coverage_grid_.info.origin.position.y = costmap_->getOriginY();
  coverage_grid_.info.origin.position.z = 0.0;
  coverage_grid_.info.origin.orientation.w = 1.0;

  // 初始化为 0 (未覆盖)
  coverage_grid_.data.assign(coverage_grid_.info.width * coverage_grid_.info.height, 0);
  
  RCLCPP_INFO(rclcpp::get_logger("SACPlanner"), "Coverage grid initialized: %dx%d", 
              coverage_grid_.info.width, coverage_grid_.info.height);
}

void SACPlanner::updateCoverageGrid(const geometry_msgs::msg::PoseStamped & pose)
{
  if (!costmap_) return;

  unsigned int mx, my;
  if (costmap_->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
    // 标记当前位置周围的区域为已覆盖
    // 作业宽度由参数 work_width 决定
    int cell_radius = static_cast<int>(work_width_ / costmap_->getResolution() / 2.0);
    
    for (int i = -cell_radius; i <= cell_radius; ++i) {
      for (int j = -cell_radius; j <= cell_radius; ++j) {
        int nx = static_cast<int>(mx) + i;
        int ny = static_cast<int>(my) + j;
        
        if (nx >= 0 && nx < (int)coverage_grid_.info.width && ny >= 0 && ny < (int)coverage_grid_.info.height) {
          coverage_grid_.data[ny * coverage_grid_.info.width + nx] = 50; // 已覆盖标记为 50
        }
      }
    }
    
    // 发布覆盖地图用于可视化
    coverage_grid_.header.stamp = node_->now();
    coverage_pub_->publish(coverage_grid_);

    // 记录并发布已过路径
    passed_path_.header = coverage_grid_.header;
    passed_path_.poses.push_back(pose);
    if (passed_path_.poses.size() > 5000) passed_path_.poses.erase(passed_path_.poses.begin());
    passed_path_pub_->publish(passed_path_);

    // 计算覆盖率百分比并发布状态
    agricultural_msgs::msg::CoverageStatus status;
    status.header.stamp = node_->now();
    int covered_cells = 0;
    for (auto val : coverage_grid_.data) {
      if (val == 50) covered_cells++;
    }
    status.covered_area = covered_cells * std::pow(coverage_grid_.info.resolution, 2);
    status.total_area = coverage_grid_.data.size() * std::pow(coverage_grid_.info.resolution, 2);
    status.coverage_percentage = (status.covered_area / status.total_area) * 100.0;
    
    // 更新缓存供getSacPrediction使用（避免重复计算和数据竞争）
    latest_coverage_status_ = status;
    
    coverage_status_pub_->publish(status);
  }
}

void SACPlanner::resetCoverageCallback(
  const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
  std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/)
{
  RCLCPP_INFO(node_->get_logger(), "🔄 Resetting Coverage Grid!");
  
  std::lock_guard<std::mutex> lock(coverage_mutex_);
  
  // 重置覆盖网格
  initCoverageGrid();
  
  // 清空已走路径
  passed_path_.poses.clear();
  passed_path_pub_->publish(passed_path_);
  
  // 重置状态
  agricultural_msgs::msg::CoverageStatus status;
  status.header.stamp = node_->now();
  status.coverage_percentage = 0.0;
  status.covered_area = 0.0;
  status.total_area = coverage_grid_.data.size() * std::pow(coverage_grid_.info.resolution, 2);
  latest_coverage_status_ = status;
  coverage_status_pub_->publish(status);
}

}  // namespace sac_planner
