#!/usr/bin/env python3
"""
Circle motion demo for xarm7.

Moves the end-effector in a circle in the Y-Z plane at a fixed X distance.
Uses the xarm_planner services (xarm_straight_plan + xarm_exec_plan) so
MoveIt2 handles collision checking and trajectory smoothing.

Prerequisites (run first in separate terminals):
  Fake/simulation:
    ros2 launch testapp circle_motion_fake.launch.py
  Real robot:
    ros2 launch testapp circle_motion_realmove.launch.py robot_ip:=<IP>
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import PlanSingleStraight, PlanPose, PlanJoint, PlanExec, SetInt16
from testapp.scene_manager import SceneManager


# --- Initial joint angles (radians) for the home/start pose ---
# The robot moves here first (joint-space OMPL) before approaching the circle.
# Set to None to skip and approach from wherever the robot currently is.
# xarm7 has 7 joints. All zeros = upright position.
angles_deg = [0, -45, 0, 45, 0, 90, 0]
INITIAL_JOINT_ANGLES = [math.radians(x) for x in angles_deg]
# --- Circle parameters (all in metres) ---
CIRCLE_CENTER_X = 0.30   # fixed X (forward from base)
CIRCLE_CENTER_Y = 0.00   # Y centre of circle
CIRCLE_CENTER_Z = 0.30   # Z centre of circle
CIRCLE_RADIUS   = 0.08   # radius
N_WAYPOINTS     = 36     # number of points (10° steps)

# Collision sensitivity for the real robot (1=least sensitive, 5=most sensitive)
# The hardware stops automatically when a collision is detected at this level.
# Has no effect in fake/simulation mode.
COLLISION_SENSITIVITY = 5

# Orientation: gripper pointing straight down (-Z global)
# RPY(pi, 0, 0) -> quaternion (x=1, y=0, z=0, w=0)
ORIENT_X = 1.0
ORIENT_Y = 0.0
ORIENT_Z = 0.0
ORIENT_W = 0.0


def make_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = ORIENT_X
    pose.orientation.y = ORIENT_Y
    pose.orientation.z = ORIENT_Z
    pose.orientation.w = ORIENT_W
    return pose


def generate_circle_waypoints() -> list[Pose]:
    """Return N_WAYPOINTS poses equally spaced around the circle."""
    waypoints = []
    for i in range(N_WAYPOINTS):
        angle = 2.0 * math.pi * i / N_WAYPOINTS
        y = CIRCLE_CENTER_Y + CIRCLE_RADIUS * math.cos(angle)
        z = CIRCLE_CENTER_Z + CIRCLE_RADIUS * math.sin(angle)
        waypoints.append(make_pose(CIRCLE_CENTER_X, y, z))
    return waypoints


class CircleMotionNode(Node):
    def __init__(self):
        super().__init__('circle_motion')

        self._straight_plan_client = self.create_client(PlanSingleStraight, 'xarm_straight_plan')
        self._pose_plan_client = self.create_client(PlanPose, 'xarm_pose_plan')
        self._joint_plan_client = self.create_client(PlanJoint, 'xarm_joint_plan')
        self._exec_client = self.create_client(PlanExec, 'xarm_exec_plan')
        self._collision_client = self.create_client(SetInt16, '/xarm/set_collision_sensitivity')

        self.get_logger().info('Waiting for xarm_planner services...')
        self._straight_plan_client.wait_for_service()
        self._pose_plan_client.wait_for_service()
        self._joint_plan_client.wait_for_service()
        self._exec_client.wait_for_service()
        self.get_logger().info('Services ready.')

        # Safety: monitor robot error state published by the real robot driver.
        # err != 0 means the hardware detected a collision or fault and stopped.
        # In fake/simulation mode this topic is never published, so _robot_error
        # stays 0 and motion proceeds normally.
        self._robot_error: int = 0
        self.create_subscription(RobotMsg, '/xarm/robot_states', self._robot_state_cb, 10)

        # Set collision sensitivity on the real robot (no-op in simulation).
        # 2-second timeout so we don't stall startup when in simulation mode.
        if self._collision_client.wait_for_service(timeout_sec=2.0):
            req = SetInt16.Request()
            req.data = COLLISION_SENSITIVITY
            self._collision_client.call_async(req)
            self.get_logger().info(f'Collision sensitivity set to {COLLISION_SENSITIVITY}/5')
        else:
            self.get_logger().info('set_collision_sensitivity not available (simulation mode)')

        self._waypoints = generate_circle_waypoints()
        self.get_logger().info(
            f'Circle: centre=({CIRCLE_CENTER_X}, {CIRCLE_CENTER_Y}, {CIRCLE_CENTER_Z}) m, '
            f'radius={CIRCLE_RADIUS} m, {N_WAYPOINTS} waypoints'
        )

        # --- Obstacles ---
        # SceneManager publishes to /collision_object so MoveIt2 plans around them.
        # Edit this section to add your own shapes or STL files.
        # The planner will automatically route the arm around any added objects.
        self._scene = SceneManager(self)

        # Example: a box obstacle placed near the circle path.
        # Remove or comment out to disable.
        # self._scene.add_box(
        #     'obstacle_box',
        #     size=(0.05, 0.05, 0.20),          # 5 cm × 5 cm × 20 cm pillar
        #     position=(- CIRCLE_CENTER_X*0.5,
        #               CIRCLE_CENTER_Y,
        #               CIRCLE_CENTER_Z*1.2),         # just outside the circle edge
        # )

        # Example: load an STL file.  Uncomment and set the path to use it.
        # self._scene.add_mesh(
        #     'my_obstacle',
        #     stl_path='/home/juhyeon/sdl_ws/src/testapp/meshes/obstacle.stl',
        #     position=(0.35, 0.0, 0.35),
        #     scale=0.001,   # use 0.001 if the STL is in millimetres
        # )

    # ------------------------------------------------------------------
    # Safety callback
    # ------------------------------------------------------------------

    def _robot_state_cb(self, msg: RobotMsg) -> None:
        if msg.err != 0 and self._robot_error == 0:
            self.get_logger().error(
                f'Robot error detected (code {msg.err}) — stopping motion.'
            )
        self._robot_error = msg.err

    # ------------------------------------------------------------------
    # Service helpers (synchronous spin_until_future_complete wrappers)
    # ------------------------------------------------------------------

    def _plan_to_pose_joint(self, pose: Pose) -> bool:
        """Joint-space planning (OMPL). Robust for large motions from any start."""
        req = PlanPose.Request()
        req.target = pose
        future = self._pose_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_pose_plan call failed (no response)')
            return False
        success = future.result().success
        if not success:
            self.get_logger().warn('xarm_pose_plan returned success=False')
        return success

    def _plan_to_joints(self, angles_rad: list) -> bool:
        """Plan to a specific joint configuration (radians). Good for homing."""
        req = PlanJoint.Request()
        req.target = [float(a) for a in angles_rad]
        future = self._joint_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_joint_plan call failed (no response)')
            return False
        success = future.result().success
        if not success:
            self.get_logger().warn('xarm_joint_plan returned success=False')
        return success

    def _plan_to_pose_straight(self, pose: Pose) -> bool:
        """Cartesian straight-line planning. Used for circle arc segments."""
        req = PlanSingleStraight.Request()
        req.target = pose
        future = self._straight_plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_straight_plan call failed (no response)')
            return False
        success = future.result().success
        if not success:
            self.get_logger().warn('xarm_straight_plan returned success=False')
        return success

    def _execute_plan(self, wait: bool = True) -> bool:
        req = PlanExec.Request()
        req.wait = wait
        future = self._exec_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('xarm_exec_plan call failed (no response)')
            return False
        success = future.result().success
        if not success:
            self.get_logger().warn('xarm_exec_plan returned success=False')
        return success

    # ------------------------------------------------------------------
    # Main motion loop
    # ------------------------------------------------------------------

    def run_circle(self, num_laps: int = 1):
        """Execute `num_laps` full circles. Pass num_laps=0 for infinite."""
        # --- Home: move to a known joint configuration before starting. ---
        # This ensures a predictable starting pose regardless of where the robot
        # was left. Set INITIAL_JOINT_ANGLES = None to skip.
        if INITIAL_JOINT_ANGLES is not None:
            self.get_logger().info(f'Moving to home position: {INITIAL_JOINT_ANGLES}')
            if not self._plan_to_joints(INITIAL_JOINT_ANGLES):
                self.get_logger().error('Home planning failed, aborting.')
                return
            if not self._execute_plan(wait=True):
                self.get_logger().error('Home execution failed, aborting.')
                return

        # --- Approach: move to the first waypoint via joint-space planning.
        # This works regardless of where the robot currently is, because OMPL
        # finds a collision-free joint path without requiring a straight
        # Cartesian line from the current position.
        self.get_logger().info('Approaching start of circle (joint-space plan)...')
        if not self._plan_to_pose_joint(self._waypoints[0]):
            self.get_logger().error('Approach planning failed, aborting.')
            return
        if not self._execute_plan(wait=True):
            self.get_logger().error('Approach execution failed, aborting.')
            return

        lap = 0
        while rclpy.ok():
            if self._robot_error != 0:
                self.get_logger().error('Robot error active, stopping circle motion.')
                return
            self.get_logger().info(f'Starting lap {lap + 1}')
            for idx, pose in enumerate(self._waypoints):
                if self._robot_error != 0:
                    self.get_logger().error('Robot error detected mid-lap, aborting.')
                    return
                self.get_logger().info(
                    f'  Waypoint {idx + 1}/{N_WAYPOINTS}: '
                    f'y={pose.position.y:.3f} z={pose.position.z:.3f}'
                )
                if not self._plan_to_pose_straight(pose):
                    self.get_logger().error('Planning failed, aborting.')
                    return
                if not self._execute_plan(wait=True):
                    self.get_logger().error('Execution failed, aborting.')
                    return

            lap += 1
            if num_laps > 0 and lap >= num_laps:
                break

        self.get_logger().info('Circle motion complete.')


def main(args=None):
    rclpy.init(args=args)
    node = CircleMotionNode()
    try:
        node.run_circle(num_laps=0)   # change to 0 for infinite loop
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
