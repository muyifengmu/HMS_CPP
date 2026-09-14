import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os

#创建一个launch文件，同时启动四个动作
def generate_launch_description():
	urdf_package_path=get_package_share_directory("robot_description") #1.获取功能包的编译路径
	default_urdf_path=os.path.join(urdf_package_path,"urdf","robot","fishbot.urdf.xacro") #2.组成完整的编译后机器人配置文件路径
	default_rviz_config_path=os.path.join(urdf_package_path,"config","display_robot_model.rviz")
	#1.动作1 声明一个可被修改的机器人配置urdf文件路径
	action_declare_arg_mode_path=launch.actions.DeclareLaunchArgument(name="model",default_value=str(default_urdf_path),description="加载的模型文件路径")

	#读取由"model"参数指定的文件内容
	substitutions_command_result=launch.substitutions.Command(["xacro ",launch.substitutions.LaunchConfiguration("model")]) #获取名为"model"的launch配置参数的值 执行shell命令xacro解析该文件
	robot_description_value=launch_ros.parameter_descriptions.ParameterValue(substitutions_command_result,value_type=str) #将文件中的机器人参数读取并赋值
	#2.动作2 启动robot_state_publisher节点 发布机器人的状态配置信息
	action_robot_state_publisher=launch_ros.actions.Node(
		package="robot_state_publisher",
		executable="robot_state_publisher",
		parameters=[{"robot_description":robot_description_value}]
	)

	#3.动作3 启动joint_state_publisher节点 发布机器人的关节配置信息
	action_joint_state_publisher=launch_ros.actions.Node(
		package="joint_state_publisher",
		executable="joint_state_publisher",
		parameters=[{"robot_description":robot_description_value}]
	)
	
	#4.动作4 启动rviz2
	action_rviz_node=launch_ros.actions.Node(
		package="rviz2",
		executable="rviz2",
		arguments=["-d",default_rviz_config_path] #添加为运行时参数 以便自动加载保存的rviz文件 和parameters是两种不同的参数书写方式 效果类似
	)

	return launch.LaunchDescription([action_declare_arg_mode_path,
								 	 action_robot_state_publisher,
								 	 action_joint_state_publisher,
								 	 action_rviz_node])