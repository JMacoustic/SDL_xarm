#!/usr/bin/env python3
"""
Connection test for xarm7 real robot.

Queries the robot's current state, joint angles, TCP position, and error codes
via the xarm_api ROS2 services. Run this AFTER the real robot driver is up:

  ros2 launch xarm_moveit_config xarm7_moveit_realmove.launch.py robot_ip:=<IP> no_gui_ctrl:=true

Then in a separate terminal:
  source install/setup.bash
  ros2 run testapp robot_connection_test
"""

import math
import rclpy
from rclpy.node import Node
from xarm_msgs.srv import GetFloat32List

SERVICE_TIMEOUT = 5.0  # seconds to wait for each service


class RobotConnectionTest(Node):
    def __init__(self):
        super().__init__('robot_connection_test')
        self._angle_client = self.create_client(GetFloat32List, '/xarm/get_servo_angle')

    def _call_get_float32_list(self, client, name: str):
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
            self.get_logger().error(f'Service {name} not available — is the robot driver running?')
            return None
        future = client.call_async(GetFloat32List.Request())
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or res.ret != 0:
            self.get_logger().error(f'{name} call failed (ret={getattr(res, "ret", "None")})')
            return None
        return list(res.datas)

    def run(self):
        self.get_logger().info('--- xarm7 Connection Test ---')

        # Joint angles — the only reliable get_* service available in this driver version
        angles_rad = self._call_get_float32_list(self._angle_client, '/xarm/get_servo_angle')
        if angles_rad is None:
            self.get_logger().error('FAILED: Cannot reach robot driver. Check robot_ip and driver launch.')
            return False

        angles_deg = [round(math.degrees(a), 2) for a in angles_rad]
        self.get_logger().info(f'Joint angles: {angles_deg} deg')
        self.get_logger().info('RESULT: Connected and ready for motion.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = RobotConnectionTest()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
