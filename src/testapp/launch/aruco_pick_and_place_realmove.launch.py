#!/usr/bin/env python3
"""
Launch the full planner stack AND the aruco_pick_and_place node together.

Usage:
  ros2 launch testapp aruco_pick_and_place_realmove.launch.py robot_ip:=<IP>

WARNING: Connects to the REAL robot. Clear the workspace and have the
         E-stop within reach before running.
"""

from launch import LaunchDescription
from launch.actions import (
    OpaqueFunction,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration('robot_ip')

    moveit_stack = IncludeLaunchDescription(
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

    # Delay the motion node so the planner services are ready before it starts.
    motion_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='testapp',
                executable='aruco_pick_and_place',
                name='aruco_pick_and_place',
                output='screen',
            )
        ],
    )

    return [moveit_stack, motion_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_ip',
            description='IP address of the xarm7 controller',
        ),
        OpaqueFunction(function=launch_setup),
    ])
