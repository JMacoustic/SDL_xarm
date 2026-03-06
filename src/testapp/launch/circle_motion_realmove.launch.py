#!/usr/bin/env python3
"""
All-in-one launch for circle_motion demo with a real xarm7.

What gets started:
  - xarm7 real robot driver (ros2_control + xarm_api)
  - move_group         (MoveIt2 motion planning)
  - xarm_planner_node  (exposes xarm_straight_plan / xarm_exec_plan services)
  - RViz2              (visualise the robot)
  - circle_motion      (our Python node, delayed 20 s to let everything settle)

Usage:
  ros2 launch testapp circle_motion_realmove.launch.py robot_ip:=192.168.1.xxx

WARNING: This moves the REAL robot. Clear the workspace and have the
         E-stop within reach before running.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration('robot_ip')

    # ------------------------------------------------------------------
    # MoveIt + real robot driver + xarm_planner_node + RViz2
    # (no_gui_ctrl=true starts xarm_planner_node automatically)
    # ------------------------------------------------------------------
    moveit_realmove_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_moveit_config'),
                'launch', '_robot_moveit_realmove.launch.py'
            ])
        ),
        launch_arguments={
            'robot_ip':    robot_ip,
            'dof':         '7',
            'robot_type':  'xarm',
            'no_gui_ctrl': 'true',
            'add_gripper': 'true',
        }.items(),
    )

    circle_motion_node = Node(
        name='circle_motion',
        package='testapp',
        executable='circle_motion',
        output='screen',
    )

    return [
        moveit_realmove_launch,
        TimerAction(period=5.0, actions=[circle_motion_node]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', description='IP address of the xarm7 controller'),
        OpaqueFunction(function=launch_setup),
    ])
