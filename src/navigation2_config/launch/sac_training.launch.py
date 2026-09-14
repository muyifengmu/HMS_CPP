import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    # 路径定义
    nav_config_dir = get_package_share_directory('navigation2_config')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map = LaunchConfiguration('map', default=os.path.join(nav_config_dir, 'maps', 'room.yaml'))
    json_path = LaunchConfiguration('json_path', default='')
    mode = LaunchConfiguration('mode', default='inference') # train or inference
    skip_phase1 = LaunchConfiguration('skip_phase1', default='false')
    use_heuristic = LaunchConfiguration('use_heuristic', default='false')
    stop_after_phase1 = LaunchConfiguration('stop_after_phase1', default='false')
    curriculum_name = LaunchConfiguration('curriculum_name', default='')
    target_exit_coverage = LaunchConfiguration('target_exit_coverage', default='98.0')
    nav2_param_path = os.path.join(nav_config_dir, 'config', 'nav2_params.yaml')
    # 自定义 HUD RViz 配置文件路径
    custom_rviz = '/home/cmh/MyProject/HMS_CPP/src/navigation2_config/rviz/hms_view.rviz'
    if os.path.exists(custom_rviz):
        rviz_config_path = custom_rviz
    else:
        rviz_config_path = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    # 【弃用 AMCL】使用 P3D 物理真值直接提供定位
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map}, {'use_sim_time': use_sim_time}]
    )

    lifecycle_manager_map = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    nav2_navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time, 'params_file': nav2_param_path}.items(),
    )

    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    p3d_tf_node = Node(
        package='sac_python_bridge', executable='p3d_tf_broadcaster', name='p3d_tf_broadcaster',
        output='screen'
    )
    
    dqn_server_node = Node(
        package='sac_python_bridge', executable='sac_server_grid', name='dqn_server',
        parameters=[{'use_sim_time': use_sim_time, 'mode': mode, 'curriculum_name': curriculum_name}], output='screen'
    )
    
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config_path], parameters=[{'use_sim_time': use_sim_time}], output='screen'
    )
    
    # automated_training.py 会在每次 reset 时发布 initialpose 给 AMCL
    trainer_node = Node(
        package='sac_python_bridge', executable='automated_training', name='automated_trainer',
        parameters=[{'use_sim_time': use_sim_time, 'json_path': json_path, 'mode': mode, 'skip_phase1': skip_phase1, 'use_heuristic': use_heuristic, 'stop_after_phase1': stop_after_phase1, 'target_exit_coverage': target_exit_coverage}], output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value=os.path.join(nav_config_dir, 'maps', 'room.yaml')),
        DeclareLaunchArgument('json_path', default_value=''),
        DeclareLaunchArgument('mode', default_value='inference'),
        DeclareLaunchArgument('skip_phase1', default_value='false'),
        DeclareLaunchArgument('use_heuristic', default_value='false'),
        DeclareLaunchArgument('stop_after_phase1', default_value='false'),
        DeclareLaunchArgument('curriculum_name', default_value=''),
        DeclareLaunchArgument('target_exit_coverage', default_value='98.0'),
        map_server_node,
        lifecycle_manager_map,
        nav2_navigation_launch,
        static_tf_map_odom,
        p3d_tf_node,
        dqn_server_node,
        rviz_node,
        trainer_node,
        # 【关键】trainer 退出后自动关闭整个 launch，让 curriculum_master 能继续下一轮
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=trainer_node,
                on_exit=[Shutdown(reason='Trainer node exited, shutting down launch.')],
            )
        ),
    ])
