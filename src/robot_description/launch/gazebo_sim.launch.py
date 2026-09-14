import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os
import json

def launch_setup(context, *args, **kwargs):
    # 1. 动态获取从命令行传入的 world 绝对路径
    world_path = launch.substitutions.LaunchConfiguration('world').perform(context)
    
    # 获取用户可能手动覆盖的坐标（如果有的话）
    x_override = launch.substitutions.LaunchConfiguration('x_pose').perform(context)
    y_override = launch.substitutions.LaunchConfiguration('y_pose').perform(context)
    yaw_override = launch.substitutions.LaunchConfiguration('yaw_pose').perform(context)
    
    # 默认为空，准备读取 JSON 里的安全点
    x_val, y_val, yaw_val = x_override, y_override, yaw_override
    
    # 如果用户没有输入覆盖指令（仍为默认的 "auto"），则智能解析 JSON
    if x_val == 'auto' and world_path.endswith('.world'):
        json_path = world_path.replace('.world', '_data.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'robot_pose' in data and data['robot_pose'] is not None:
                        pos, yaw = data['robot_pose']
                        x_val, y_val, yaw_val = str(pos[0]), str(pos[1]), str(yaw)
                        print(f"🚀 [Launch Parser] 自动从 JSON 提取到绝对安全起始点: X={x_val}, Y={y_val}, Yaw={yaw_val}")
            except Exception as e:
                print(f"⚠️ [Launch Parser] 解析 JSON 失败: {e}，将回退到原点 0.0")
                x_val, y_val, yaw_val = '0.0', '0.0', '0.0'
        else:
            print(f"⚠️ [Launch Parser] 找不到对应的 {json_path}，将回退到原点 0.0")
            x_val, y_val, yaw_val = '0.0', '0.0', '0.0'
    elif x_val == 'auto':
        x_val, y_val, yaw_val = '0.0', '0.0', '0.0'

    # 读取 URDF 模型
    urdf_package_path = get_package_share_directory("robot_description")
    default_urdf_path = os.path.join(urdf_package_path, "urdf", "robot", "fishbot.urdf.xacro")
    
    # 这里有点绕，因为 xacro 调用必须返回字符串
    substitutions_command_result = launch.substitutions.Command(["xacro ", str(default_urdf_path)])
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(substitutions_command_result, value_type=str)
    
    action_robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description_value}]
    )

    action_include_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            [get_package_share_directory("gazebo_ros"), "/launch", "/gazebo.launch.py"]),
        launch_arguments=[("world", world_path), ("verbose", "true")]
    )

    action_spawn_entity = launch_ros.actions.Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "fishbot",
            "-topic", "robot_description",
            "-x", x_val,
            "-y", y_val,
            "-Y", yaw_val,
        ]
    )

    action_load_joint_state_controller = launch.actions.ExecuteProcess(
        cmd="ros2 control load_controller fishbot_joint_state_broadcaster --set-state active".split(" "),
        output="screen"
    )

    action_load_diff_drive_controller = launch.actions.ExecuteProcess(
        cmd="ros2 control load_controller fishbot_diff_drive_controller --set-state active".split(" "),
        output="screen"
    )

    return [
        action_robot_state_publisher,
        action_include_launch,
        action_spawn_entity,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=action_spawn_entity,
                on_exit=[action_load_joint_state_controller],
            )
        ),
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=action_load_joint_state_controller,
                on_exit=[action_load_diff_drive_controller],
            )
        ),
    ]

def generate_launch_description():
    world_package_path = get_package_share_directory("agricultural_world")
    default_gazebo_world_path = os.path.join(world_package_path, "world", "field.world")

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name="world", default_value=str(default_gazebo_world_path), description="加载的世界文件路径"),
        launch.actions.DeclareLaunchArgument(name='x_pose', default_value='auto'),
        launch.actions.DeclareLaunchArgument(name='y_pose', default_value='auto'),
        launch.actions.DeclareLaunchArgument(name='yaw_pose', default_value='auto'),
        launch.actions.OpaqueFunction(function=launch_setup)
    ])