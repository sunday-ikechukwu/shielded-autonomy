# full_system.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    pkg = get_package_share_directory('shielded_autonomy')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup') # Needed for localization


    map_file    = os.path.join(pkg, 'maps',   'world_map.yaml')
    params_file = os.path.join(pkg, 'config', 'nav2_params_mppi.yaml')

    rviz_config_dir = os.path.join(get_package_share_directory('turtlebot3_navigation2'), 'rviz', 'tb3_navigation2.rviz')

    # Launch world 
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'turtlebot3_world.launch.py')  # your existing world launch
        )
    )

    # Launch Localization (Map Server + AMCL) ── THE MISSING PIECE
    localization_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_pkg, 'launch', 'localization_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'map': map_file,
                    'params_file': params_file,
                    'autostart': 'true'
                }.items()
            )
        ]
    )

    # Launch Nav2 after world is ready
    nav2_launch = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg, 'launch', 'navigation2_bringup_custom.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'map': map_file,
                    'params_file': params_file,
                    'autostart': 'true'
                }.items()
            )
        ]
    )

    rviz_node  = TimerAction(
        period = 12.0,
        actions = [
            Node(
                package    = 'rviz2',
                executable = 'rviz2',
                name       = 'rviz2',
                arguments  = ['-d', rviz_config_dir],
                parameters = [{'use_sim_time': use_sim_time}],
                output     = 'screen'
            )
        ]
    )

    return LaunchDescription([
        world_launch,
        localization_launch,
        nav2_launch,
        rviz_node
    ])