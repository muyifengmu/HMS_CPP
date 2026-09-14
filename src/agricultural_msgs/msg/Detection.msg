# 单个检测目标的描述
std_msgs/Header header

# 目标类别信息
string class_name
float32 confidence

# 边界框信息（像素坐标）
geometry_msgs/Point top_left     # 左上角点
geometry_msgs/Point bottom_right # 右下角点

# 世界坐标系中的位置（如果已知）
geometry_msgs/Point world_position
float32 distance                 # 到机器人的距离

# 农业专用属性
int8 crop_type                   # 作物类型 (0:未知, 1:小麦, 2:玉米, 3:水稻)
bool is_obstacle                 # 是否为障碍物
float32 obstacle_risk_level      # 障碍风险等级 (0-1)