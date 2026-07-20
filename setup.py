from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'shielded_autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add worlds folder
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        # Add all launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        #add rviz_config
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/bc_model', glob('bc_model/*')),
        ('share/' + package_name + '/bc_model_v2', glob('bc_model_v2/*')),
        ('share/' + package_name + '/dagger_model', glob('dagger_model/*')),
        ('share/' + package_name + '/dagger_model_V2', glob('dagger_model_V2/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='precious_weal',
    maintainer_email='ikechukwusundday5699@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'metrics_recorder        = shielded_autonomy.metrics_recorder:main',
            'imitation_data_recorder = shielded_autonomy.imitation_data_recorder:main',
            'bc_inference_node       = shielded_autonomy.bc_inference_node:main',
            'bc_vs_mppi_logger       = shielded_autonomy.bc_vs_mppi_logger:main',
            'safety_monitor_node     = shielded_autonomy.safety_monitor_node:main',
            'arbitration_node        = shielded_autonomy.arbitration_node:main',
            'safe_dagger_recorder_node = shielded_autonomy.safe_dagger_recorder_node:main',
            'safe_dagger_inference_node = shielded_autonomy.safe_dagger_inference_node:main',

            'safe_dagger_recorder_node_v2 = shielded_autonomy.safe_dagger_recorder_node_v2:main',
            'imitation_data_recorder_v2 = shielded_autonomy.imitation_data_recorder_v2:main',
            'bc_inference_node_v2 = shielded_autonomy.bc_inference_node_v2:main',
            'safety_monitor_node_v2 = shielded_autonomy.safety_monitor_node_v2:main',
            'arbitration_node_v2 = shielded_autonomy.arbitration_node_v2:main',
            'policy_metrics_node = shielded_autonomy.policy_metrics_node:main',
            'runtime_metrics_node = shielded_autonomy.runtime_metrics_node:main',
        ],
    },
)
