from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    bc_inference_launch = Node(
        package= 'shielded_autonomy',
        executable= 'bc_inference_node',
        name= 'bc_inference_node',
        output= 'screen',
        emulate_tty= True,
    )


    safety_monitor_launch = Node(
        package= 'shielded_autonomy',
        executable= 'safety_monitor_node',
        name= 'safety_monitor_node',
        output= 'screen',
        emulate_tty= True
    )

    arbitration_launch = Node(
        package= 'shielded_autonomy',
        executable= 'arbitration_node',
        name= 'arbitration_node',
        output= 'screen',
        emulate_tty= True    
    )

    safe_dagger_recorder_launch = Node(
        package='shielded_autonomy',
        executable='safe_dagger_recorder_node',
        name='safe_dagger_recorder_node',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        bc_inference_launch,
        safety_monitor_launch,
        arbitration_launch,
        safe_dagger_recorder_launch
    ])