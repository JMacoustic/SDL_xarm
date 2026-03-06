#!/usr/bin/env python3
"""
All-in-one launch for circle_motion demo in simulation (fake controllers).

What gets started (all from a single command):
  - fake ros2_control  (simulates joint motion without a real robot)
  - move_group         (MoveIt2 motion planning)
  - xarm_planner_node  (exposes xarm_straight_plan / xarm_exec_plan services)
  - RViz2              (visualise the robot)
  - circle_motion      (our Python node, delayed 15 s to let everything settle)

Usage:
  ros2 launch testapp circle_motion_fake.launch.py
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # ------------------------------------------------------------------
    # MoveIt + fake robot + xarm_planner_node + RViz2
    #
    # no_gui_ctrl=true does three things:
    #   1. Launches RViz with the "planner" layout (no interactive marker panel)
    #   2. Starts xarm_planner_node (provides xarm_straight_plan / xarm_exec_plan)
    #   3. Skips the MoveIt motion-planning GUI panel
    # ------------------------------------------------------------------
    moveit_fake_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_moveit_config'),
                'launch', '_robot_moveit_fake.launch.py'
            ])
        ),
        launch_arguments={
            'dof':         '7',
            'robot_type':  'xarm',
            'no_gui_ctrl': 'true',
            'add_gripper': 'true',
        }.items(),
    )

    # Our node — wait for the planner services to become available
    circle_motion_node = Node(
        name='circle_motion',
        package='testapp',
        executable='circle_motion',
        output='screen',
    )

    return [
        moveit_fake_launch,
        TimerAction(period=5.0, actions=[circle_motion_node]),
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
