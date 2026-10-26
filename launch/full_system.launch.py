# full_system.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('shielded_autonomy')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup') # Needed for localization

    #Default fallback path
    default_map_file = os.path.join(pkg, 'maps', 'experiment_b_map2.yaml')
    defualt_params_file = os.path.join(pkg, 'config', 'nav2_params_mppi.yaml')
    defualt_world_file = os.path.join(pkg, 'worlds', 'experiment_b_world2.sdf')
    rviz_config_dir = os.path.join(get_package_share_directory('turtlebot3_navigation2'), 'rviz', 'tb3_navigation2.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_config = LaunchConfiguration('map')
    world_config = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw_pose = LaunchConfiguration('yaw_pose') 

    declare_x_arg = DeclareLaunchArgument(
        'x_pose',
        default_value='0.0', 
        description='X position'
    )
    declare_y_arg = DeclareLaunchArgument(
        'y_pose', 
        default_value='0.0', 
        description='Y position'
    )
    declare_yaw_arg = DeclareLaunchArgument(
        'yaw_pose', 
        default_value='0.0', 
        description='Yaw (Theta) angle in radians'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use Simulation (Gazebo) clock'
    )
    declare_map_arg= DeclareLaunchArgument(
        'map',
        default_value=default_map_file,
        description='Full path to map yaml file'
    )
    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value=defualt_world_file,
        description='Full path to world file'
    )

    # Launch world 
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'turtlebot3_world.launch.py')  # your existing world launch
        ),
        launch_arguments={'world': world_config}.items()
    )

    # Launch Localization (Map Server + AMCL)
    localization_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_pkg, 'launch', 'localization_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'map': map_config,
                    'params_file': defualt_params_file,
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
                    'map': map_config,
                    'params_file': defualt_params_file,
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
    spawn_turtlebot_cmd = Node(
    package='gazebo_ros',
    executable='spawn_entity.py',
    arguments=[
        '-x', x_pose,  # <-- use launch argument
        '-y', y_pose,  # <-- use launch argument
        '-Y', yaw_pose,
    ],
    output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_map_arg,
        declare_world_arg,
        declare_x_arg,
        declare_y_arg,
        declare_yaw_arg,
        spawn_turtlebot_cmd,
        world_launch,
        localization_launch,
        nav2_launch,
        rviz_node
    ])