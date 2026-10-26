from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    bc_inference_launch = Node(
        package= 'shielded_autonomy',
        executable= 'bc_inference_node_v2',
        name= 'bc_inference_node_v2',
        output= 'screen',
        emulate_tty= True,
    )

    safe_dagger_inference_launch = Node(
        package= 'shielded_autonomy',
        executable= 'safe_dagger_inference_node',
        name= 'safe_dagger_inference_node',
        output= 'screen',
        emulate_tty= True,
    )

    safety_monitor_launch = Node(
        package= 'shielded_autonomy',
        executable= 'safety_monitor_node_v2',
        name= 'safety_monitor_node_v2',
        output= 'screen',
        emulate_tty= True
    )

    arbitration_launch = Node(
        package= 'shielded_autonomy',
        executable= 'arbitration_node_v2',
        name= 'arbitration_node_v2',
        output= 'screen',
        emulate_tty= True    
    )

    safe_dagger_recorder_launch = Node(
        package='shielded_autonomy',
        executable='safe_dagger_recorder_node_v2',
        name='safe_dagger_recorder_node_v2',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        safe_dagger_inference_launch,
        # bc_inference_launch,
        safety_monitor_launch,
        arbitration_launch,
        # safe_dagger_recorder_launch
    ])