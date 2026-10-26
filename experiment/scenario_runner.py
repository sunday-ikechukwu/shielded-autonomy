#!/usr/bin/env python3
"""
Scenario Runner Utility
Loads a scenario from a YAML configuration and sends the goal pose to Nav2.
"""

import argparse
import math
import os
import re
import sys

import geometry_msgs.msg

# ROS 2 imports
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def yaw_to_quaternion(yaw):
    """Convert yaw angle (radians) to geometry_msgs.msg.Quaternion."""
    q = geometry_msgs.msg.Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def load_yaml(yaml_path):
    """Loads a YAML file and returns a dictionary of scenarios."""
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file '{yaml_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Failed to parse YAML file '{yaml_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        return {}

    scenarios = {}
    # Support list format: scenarios: [{id: A1, ...}]
    if 'scenarios' in data and isinstance(data['scenarios'], list):
        for item in data['scenarios']:
            if isinstance(item, dict) and 'id' in item:
                scenarios[str(item['id'])] = item

    # Support dictionary format: scenario_id: {description: ...}
    for key, value in data.items():
        if key != 'scenarios' and isinstance(value, dict):
            scenarios[str(key)] = value

    return scenarios


class ScenarioRunner(Node):
    """ROS 2 Node acting as a client for NavigateToPose."""
    def __init__(self):
        super().__init__('scenario_runner_node')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.result_future = None
        self.navigation_status = None  # SUCCESS, FAILED, or REJECTED

    def wait_for_server(self, timeout_sec=15.0):
        """Wait for the action server to become available."""
        self.get_logger().info("Connecting to 'navigate_to_pose' action server...")
        start_time = self.get_clock().now()
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().error("Action server connection timed out.")
                return False
            self.get_logger().info("Still waiting for action server...")
        return True

    def send_goal(self, x, y, yaw):
        """Send navigation goal pose to the action server."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = yaw_to_quaternion(yaw)

        self.get_logger().info("Sending navigation goal...")
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Callback when action server responds to the goal request."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was REJECTED by action server.")
            self.navigation_status = "REJECTED"
            return

        self.get_logger().info("Goal accepted by action server.")
        self.goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Callback when action execution completes."""
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.navigation_status = "SUCCESS"
        else:
            self.get_logger().warn(f"Navigation ended with status code: {status}")
            self.navigation_status = "FAILED"


def resolve_scenario(scenarios, scenario_id):
    """Finds the scenario by exact key, case-insensitive, or numeric suffix matching."""
    # 1. Exact match
    if scenario_id in scenarios:
        return scenario_id, scenarios[scenario_id]

    # 2. Case-insensitive match
    for k, v in scenarios.items():
        if k.lower() == scenario_id.lower():
            return k, v

    # 3. Numeric suffix matching (e.g. scenario_3 -> A3 or 3)
    match = re.search(r'\d+', scenario_id)
    if match:
        num = match.group(0)
        for k, v in scenarios.items():
            if k == num or k.endswith(num) or (len(k) == 2 and k[0].isalpha() and k[1] == num):
                return k, v

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Standalone ROS 2 Nav2 Scenario Runner")
    parser.add_argument(
        'scenario',
        nargs='?',
        help="Scenario ID to execute (e.g. A1, scenario_3)"
    )
    parser.add_argument(
        '--yaml',
        default='/home/precious_weal/nav2_ws/src/shielded_autonomy/experiment/experiment_B_scenarios.yaml',
        help="Path to scenarios YAML configuration file"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available scenarios in the YAML file"
    )

    args = parser.parse_args()

    # Load YAML file
    scenarios = load_yaml(args.yaml)

    # Handle listing
    if args.list:
        print("\nAvailable scenarios:")
        for name, spec in sorted(scenarios.items()):
            desc = spec.get('description', 'No description provided')
            print(f"  - {name}: {desc}")
        sys.exit(0)

    # Check for scenario argument
    if not args.scenario:
        parser.print_help()
        sys.exit(1)

    # Resolve scenario
    resolved_id, spec = resolve_scenario(scenarios, args.scenario)
    if not spec:
        print(f"Error: Unknown scenario '{args.scenario}'. Use --list to see options.", file=sys.stderr)
        sys.exit(1)

    # Extract goal pose details
    goal_spec = spec.get('goal')
    if not goal_spec or 'x' not in goal_spec or 'y' not in goal_spec:
        print(f"Error: Scenario '{resolved_id}' is missing required goal geometry.", file=sys.stderr)
        sys.exit(1)

    goal_x = goal_spec['x']
    goal_y = goal_spec['y']
    goal_yaw = goal_spec.get('yaw', 0.0)
    desc = spec.get('description', 'N/A')
    behaviour = spec.get('behaviour', 'N/A')

    # Print scenario summary exactly as requested
    print("------------------------------------------------\n")
    print(f"Scenario: {resolved_id}\n")
    print(f"Description: {desc}\n")
    print(f"Behaviour: {behaviour}\n")
    print("Goal\n")
    print(f"x: {goal_x}\n")
    print(f"y: {goal_y}\n")
    print(f"yaw: {goal_yaw}\n")
    print("------------------------------------------------\n")

    # Initialize ROS 2 communication
    try:
        rclpy.init()
    except Exception as e:
        print(f"Error: Failed to initialize rclpy: {e}. Sourced workspace?", file=sys.stderr)
        sys.exit(1)

    runner = ScenarioRunner()

    if not runner.wait_for_server():
        print("Error: Nav2 server is unavailable.", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    # Send the goal
    runner.send_goal(goal_x, goal_y, goal_yaw)

    # Spin to process callbacks
    try:
        while rclpy.ok() and runner.navigation_status is None:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nInterrupt received. Cancelling navigation...")
        if runner.goal_handle:
            cancel_fut = runner.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(runner, cancel_fut, timeout_sec=3.0)
        rclpy.shutdown()
        sys.exit(1)

    # Print final result
    status = runner.navigation_status
    if status == "SUCCESS":
        print("Navigation SUCCEEDED")
        exit_code = 0
    else:
        print("Navigation FAILED")
        exit_code = 1

    # Clean shutdown
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
