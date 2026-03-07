#!/usr/bin/env python3
"""
Base launch for xArm7 in simulation (fake controllers).

Starts: fake ros2_control + MoveIt2 + xarm_planner_node + RViz2
Does NOT start any motion node — run your node separately:
  ros2 run testapp pick_and_place
  ros2 run testapp circle_motion

Usage:
  ros2 launch testapp base_fake.launch.py
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
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
    return [moveit_fake_launch]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
