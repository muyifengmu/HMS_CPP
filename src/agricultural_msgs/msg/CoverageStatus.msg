std_msgs/Header header

# 覆盖进度
float32 coverage_percentage      # 覆盖百分比 (0-100)
float32 covered_area             # 已覆盖面积 (平方米)
float32 total_area               # 总面积 (平方米)

# 效率指标
float32 estimated_time_remaining # 预计剩余时间 (秒)
float32 average_speed            # 平均作业速度 (米/秒)
float32 path_efficiency          # 路径效率 (0-1)

# 覆盖质量
int32 repeated_coverage_count    # 重复覆盖次数
float32 missed_area              # 漏覆盖面积

# 时间戳
builtin_interfaces/Time start_time
builtin_interfaces/Time current_time