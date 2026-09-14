#pragma once

#include <string>
#include <memory>
#include <vector>
#include <mutex>

#include "nav2_core/global_planner.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "agricultural_msgs/srv/sac_predict.hpp"
#include "std_srvs/srv/empty.hpp"



namespace sac_planner
{

class SACPlanner : public nav2_core::GlobalPlanner
{
public:
  SACPlanner();
  ~SACPlanner();

  // 导航2核心接口
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  // 规划路径的核心方法
  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  // 节点指针
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D* costmap_;

  // SAC服务客户端
  rclcpp::Client<agricultural_msgs::srv::SACPredict>::SharedPtr sac_predict_client_;

  // 配置参数
  double tolerance_;
  bool use_astar_;
  bool allow_unknown_;
  double work_width_;

  // 协助覆盖逻辑的方法
  void initCoverageGrid();
  void updateCoverageGrid(const geometry_msgs::msg::PoseStamped & pose);
  
  // 覆盖相关的成员变量
  nav_msgs::msg::OccupancyGrid coverage_grid_;
  nav_msgs::msg::Path passed_path_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr coverage_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr passed_path_pub_;
  rclcpp::Publisher<agricultural_msgs::msg::CoverageStatus>::SharedPtr coverage_status_pub_;
  rclcpp::TimerBase::SharedPtr coverage_timer_;
  agricultural_msgs::msg::CoverageStatus latest_coverage_status_;  // 缓存最后的coverage状态
  std::mutex coverage_mutex_;  // 保护coverage_grid_的互斥锁

  // 辅助方法
  void initializeDqnClient();
  agricultural_msgs::srv::SACPredict::Response::SharedPtr getSacPrediction(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal);
  nav_msgs::msg::Path generatePathFromSacActions(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const agricultural_msgs::srv::SACPredict::Response::SharedPtr & prediction);

  // 重置覆盖率服务
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
  void resetCoverageCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                             std::shared_ptr<std_srvs::srv::Empty::Response> response);
};

}  // namespace sac_planner