#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time'),

        # Launch the detect_and_segment node
        Node(
            package='pallet_detection_and_segmentation',
            executable='detect_and_segment.py',
            name='detect_and_segment',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('/camera/color/image_raw', '/camera/color/image_raw'),
                        ('/camera/depth/image_raw', '/camera/depth/image_raw')]
        ),

        # Launch the video_publisher node
        Node(
            package='pallet_detection_and_segmentation',
            executable='video_publisher.py',
            name='video_publisher',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('/camera/color/image_raw', '/camera/color/image_raw')]
        ),
    ])
