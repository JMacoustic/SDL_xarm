#!/usr/bin/env python3
"""
Base launch for xArm7 with a real robot.

Starts: real robot driver + MoveIt2 + xarm_planner_node + RViz2
Does NOT start any motion node — run your node separately:
  ros2 run testapp pick_and_place
  ros2 run testapp circle_motion

Usage:
  ros2 launch testapp base_realmove.launch.py robot_ip:=192.168.1.xxx

WARNING: Connects to the REAL robot. Clear the workspace and have the
         E-stop within reach before running.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration('robot_ip')

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
    return [moveit_realmove_launch]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', description='IP address of the xarm7 controller'),
        OpaqueFunction(function=launch_setup),
    ])
