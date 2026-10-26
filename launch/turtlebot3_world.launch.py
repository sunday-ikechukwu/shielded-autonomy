
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg = get_package_share_directory('shielded_autonomy')

    default_world_file = os.path.join(
        pkg,
        'worlds',
        'research_office_world.sdf'
    )

    world_config = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
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
        description='Use simulation (Gazebo) clock if true'
    )
    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_file,
        description='Full path to world file to load'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_config}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = Node(
    package='gazebo_ros',
    executable='spawn_entity.py',
    arguments=[
        '-entity', 'waffle',
        '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf',
        '-x', x_pose,  # <-- use launch argument
        '-y', y_pose,  # <-- use launch argument
        '-z', '0.01',
        '-Y', yaw_pose,
    ],
    output='screen'
    )

    
    ld = LaunchDescription()

    ld.add_action(declare_world_arg)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_x_arg)
    ld.add_action(declare_y_arg)
    ld.add_action(declare_yaw_arg)
    ld.add_action(gzserver_cmd)
    # ld.add_action(gzclient_cmd) # Optional:  Uncomment to launch Gazebo client (GUI)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
