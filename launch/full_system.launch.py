# full_system.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg = get_package_share_directory('nav2_classical_planner')
    tb3_navigation = get_package_share_directory('turtlebot3_navigation2')

    map_file    = os.path.join(pkg, 'maps',   'world_map.yaml')
    params_file = os.path.join(pkg, 'config', 'nav2_params.yaml')

    # ── Launch your world first
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'turtlebot3_world.launch.py')  # your existing world launch
        )
    )

    # ── Launch Nav2 after world is ready
    nav2_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(tb3_navigation, 'launch', 'navigation2.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': 'true',
                    'map': map_file,
                    'params_file': params_file,
                }.items()
            )
        ]
    )

    return LaunchDescription([
        world_launch,
        nav2_launch,
    ])