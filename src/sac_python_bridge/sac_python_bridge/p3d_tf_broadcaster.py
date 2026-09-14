#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class P3D_TF_Broadcaster(Node):
    def __init__(self):
        super().__init__('p3d_tf_broadcaster')
        self.subscription = self.create_subscription(
            Odometry,
            '/ground_truth_odom',
            self.odom_callback,
            10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("🎯 P3D TF 真值广播器已启动，正在完全接管底层定位（抛弃轮式打滑误差）...")

    def odom_callback(self, msg):
        t = TransformStamped()
        
        # P3D 原生从 world -> base_footprint。
        # 在 launch 里发布零误差 static tf (map -> odom)
        # 将真实位姿灌入 (odom -> base_footprint)
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = P3D_TF_Broadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
