#!/usr/bin/env python3
"""
Pick and place demo for xArm7 with G2 gripper.

Grab and place poses are specified as SE3 (4x4 homogeneous numpy matrices):
    [[R | t],
     [0 | 1]]

The gripper approaches and departs along the Z-axis of the given SE3 frame,
so the gripper XY plane aligns with the object's XY plane at contact.

Usage:
  Fake/simulation:
    ros2 launch testapp circle_motion_fake.launch.py
  Real robot:
    ros2 launch testapp circle_motion_realmove.launch.py robot_ip:=<IP>

Then in a separate terminal:
  ros2 run testapp pick_and_place
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import PlanSingleStraight, PlanPose, PlanJoint, PlanExec, SetInt16


# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------

# Approach/depart distance along the SE3 Z-axis (metres)
APPROACH_DIST = 0.10

# Gripper joint value: 0.0 = fully closed, 0.85 = fully open
GRIPPER_OPEN   = 0.85
GRIPPER_CLOSED = 0.00

# Collision sensitivity for the real robot (1=least, 5=most)
COLLISION_SENSITIVITY = 3

# ---------------------------------------------------------------------------
# Example SE3 poses (replace with your actual values)
# Convention: Z-axis of the frame = gripper approach direction
# ---------------------------------------------------------------------------

def example_se3(x, y, z, rx=math.pi, ry=0.0, rz=0.0):
    """Build an SE3 from position and ZYX Euler angles (radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    R = np.array([
        [cy*cz,  cz*sx*sy - cx*sz,  cx*cz*sy + sx*sz],
        [cy*sz,  cx*cz + sx*sy*sz,  cx*sy*sz - cz*sx],
        [-sy,    cy*sx,              cx*cy            ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


# Grab pose: position in metres, orientation as ZYX Euler angles
GRAB_SE3  = example_se3(x=0.35, y= 0.10, z=0.20)
# Place pose
PLACE_SE3 = example_se3(x=0.35, y=-0.10, z=0.20)

# ---------------------------------------------------------------------------
# SE3 helpers
# ---------------------------------------------------------------------------

def rotation_to_quaternion(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def se3_to_pose(T: np.ndarray) -> Pose:
    """Convert 4x4 SE3 matrix to geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = rotation_to_quaternion(T[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def approach_pose(T: np.ndarray, dist: float) -> Pose:
    """
    Return a pose offset by `dist` along the -Z axis of T.
    This is the pre-grasp / post-place hover position.
    """
    z_axis = T[:3, 2]           # Z column of rotation = approach direction
    T_approach = T.copy()
    T_approach[:3, 3] -= dist * z_axis
    return se3_to_pose(T_approach)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place')

        self._straight_plan_client  = self.create_client(PlanSingleStraight, 'xarm_straight_plan')
        self._pose_plan_client      = self.create_client(PlanPose,           'xarm_pose_plan')
        self._exec_client           = self.create_client(PlanExec,           'xarm_exec_plan')
        self._gripper_plan_client   = self.create_client(PlanJoint,          'xarm_gripper_joint_plan')
        self._gripper_exec_client   = self.create_client(PlanExec,           'xarm_gripper_exec_plan')
        self._collision_client      = self.create_client(SetInt16,           '/xarm/set_collision_sensitivity')

        self.get_logger().info('Waiting for arm planner services...')
        self._straight_plan_client.wait_for_service()
        self._pose_plan_client.wait_for_service()
        self._exec_client.wait_for_service()
        self.get_logger().info('Waiting for gripper planner services...')
        self._gripper_plan_client.wait_for_service()
        self._gripper_exec_client.wait_for_service()
        self.get_logger().info('All services ready.')

        # Safety: monitor robot error state
        self._robot_error: int = 0
        self.create_subscription(RobotMsg, '/xarm/robot_states', self._robot_state_cb, 10)

        # Set collision sensitivity (real robot only)
        if self._collision_client.wait_for_service(timeout_sec=2.0):
            req = SetInt16.Request()
            req.data = COLLISION_SENSITIVITY
            self._collision_client.call_async(req)
            self.get_logger().info(f'Collision sensitivity set to {COLLISION_SENSITIVITY}/5')

    # ------------------------------------------------------------------
    # Safety callback
    # ------------------------------------------------------------------

    def _robot_state_cb(self, msg: RobotMsg) -> None:
        if msg.err != 0 and self._robot_error == 0:
            self.get_logger().error(f'Robot error (code {msg.err}) — stopping motion.')
        self._robot_error = msg.err

    def _check_error(self) -> bool:
        if self._robot_error != 0:
            self.get_logger().error('Robot error active, aborting.')
            return False
        return True

    # ------------------------------------------------------------------
    # Arm planning helpers
    # ------------------------------------------------------------------

    def _plan_pose(self, pose: Pose) -> bool:
        """OMPL pose plan — for large unconstrained moves."""
        req = PlanPose.Request()
        req.target = pose
        future = self._pose_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_pose_plan: no response')
            return False
        if not future.result().success:
            self.get_logger().warn('xarm_pose_plan: planning failed')
        return future.result().success

    def _plan_straight(self, pose: Pose) -> bool:
        """Cartesian straight-line plan — for approach/depart segments."""
        req = PlanSingleStraight.Request()
        req.target = pose
        future = self._straight_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_straight_plan: no response')
            return False
        if not future.result().success:
            self.get_logger().warn('xarm_straight_plan: planning failed')
        return future.result().success

    def _execute(self) -> bool:
        req = PlanExec.Request()
        req.wait = True
        future = self._exec_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_exec_plan: no response')
            return False
        if not future.result().success:
            self.get_logger().warn('xarm_exec_plan: execution failed')
        return future.result().success

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def _move_gripper(self, position: float) -> bool:
        """Move gripper to joint position (0.0=closed, 0.85=open)."""
        req = PlanJoint.Request()
        req.target = [float(position)]
        future = self._gripper_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None or not future.result().success:
            self.get_logger().error('Gripper plan failed')
            return False

        exec_req = PlanExec.Request()
        exec_req.wait = True
        future = self._gripper_exec_client.call_async(exec_req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None or not future.result().success:
            self.get_logger().error('Gripper execute failed')
            return False
        return True

    def _open_gripper(self) -> bool:
        self.get_logger().info('Opening gripper...')
        return self._move_gripper(GRIPPER_OPEN)

    def _close_gripper(self) -> bool:
        self.get_logger().info('Closing gripper...')
        return self._move_gripper(GRIPPER_CLOSED)

    # ------------------------------------------------------------------
    # Compound moves
    # ------------------------------------------------------------------

    def _move_to_pose(self, pose: Pose, label: str) -> bool:
        """OMPL move + execute."""
        self.get_logger().info(f'Moving to {label} (OMPL)...')
        if not self._check_error(): return False
        if not self._plan_pose(pose): return False
        if not self._execute(): return False
        return True

    def _move_straight_to(self, pose: Pose, label: str) -> bool:
        """Straight-line Cartesian move + execute."""
        self.get_logger().info(f'Straight move to {label}...')
        if not self._check_error(): return False
        if not self._plan_straight(pose): return False
        if not self._execute(): return False
        return True

    # ------------------------------------------------------------------
    # Pick and place sequence
    # ------------------------------------------------------------------

    def run(self, grab_se3: np.ndarray, place_se3: np.ndarray):
        """
        Execute one pick-and-place cycle.

        Args:
            grab_se3:  4x4 SE3 of the grasp pose (Z-axis = approach direction)
            place_se3: 4x4 SE3 of the release pose (Z-axis = approach direction)
        """
        pre_grasp  = approach_pose(grab_se3,  APPROACH_DIST)
        grasp      = se3_to_pose(grab_se3)
        pre_place  = approach_pose(place_se3, APPROACH_DIST)
        place      = se3_to_pose(place_se3)

        self.get_logger().info('=== Pick and Place Start ===')

        # 1. Open gripper before approaching
        if not self._open_gripper(): return

        # 2. OMPL move to pre-grasp hover
        if not self._move_to_pose(pre_grasp, 'pre-grasp'): return

        # 3. Straight-line approach to grasp pose
        if not self._move_straight_to(grasp, 'grasp'): return

        # 4. Close gripper
        if not self._close_gripper(): return

        # 5. Straight-line depart back to pre-grasp
        if not self._move_straight_to(pre_grasp, 'post-grasp hover'): return

        # 6. OMPL move to pre-place hover
        if not self._move_to_pose(pre_place, 'pre-place'): return

        # 7. Straight-line descend to place pose
        if not self._move_straight_to(place, 'place'): return

        # 8. Open gripper to release
        if not self._open_gripper(): return

        # 9. Straight-line depart back to pre-place
        if not self._move_straight_to(pre_place, 'post-place hover'): return

        self.get_logger().info('=== Pick and Place Complete ===')


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    try:
        node.run(grab_se3=GRAB_SE3, place_se3=PLACE_SE3)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
