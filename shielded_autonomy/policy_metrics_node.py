"""
policy_metrics_node.py

ROS 2 Humble Python node to evaluate the performance of a navigation policy
without expert intervention during policy evaluation experiments.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
import math
import csv
import os

class PolicyMetricsNode(Node):
    def __init__(self):
        super().__init__('policy_metrics_node')

        # --- Parameters ---
        self.declare_parameter('policy_name', 'mppi')
        self.declare_parameter('environment', 'unknown_environment')
        self.declare_parameter('scenario', 'scenario_1')
        self.declare_parameter('run_number', 1)
        self.declare_parameter('output_directory', '~/nav2_ws/src/shielded_autonomy/results/experiment_B')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)# Ensure simulation clock is used by default

        # --- Constants ---
        self.COLLISION_THRESHOLD = 0.24  # Reused from metrics_recorder.py
        self.NEAR_MISS_THRESHOLD = 0.35  # Reused from metrics_recorder.py (local_costmap inflation radius)
        self.NEAR_MISS_HYSTERESIS = 0.05 # 5cm hysteresis to prevent near-miss chattering
        
        # Stuck and Timeout limits (in simulation seconds)
        self.STUCK_TIMEOUT_LIMIT = 60.0    
        self.ABSOLUTE_TIMEOUT_LIMIT = 300.0 

        # --- State Variables ---
        self.recording_active = True
        self.csv_written = False
        self.tracked_goal_id = None
        self.in_collision = False
        self.in_near_miss = False

        self.x = 0.0
        self.y = 0.0
        self.prev_x = None
        self.prev_y = None

        self.actual_path_length = 0.0
        self.planned_path_length = 0.0

        # Velocity tracking
        self.linear_velocities = []
        self.angular_velocities = []

        # Smoothness tracking
        self.prev_linear_vel = None
        self.prev_angular_vel = None
        self.velocity_change = []

        # Safety tracking
        self.min_obstacle_distance = float('inf')
        self.collision_count = 0
        self.near_miss_count = 0

        # Timing & Stuck tracking
        self.mission_start_time = None
        self.last_navigation_time = 0.0
        self.last_progress_time = 0.0
        self.previous_progress_pose = None

        # --- QoS for sensor topics ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=sensor_qos)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)
        self.feedback_sub = self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )

        # --- Periodic loop to monitor timeout ---
        self.timer = self.create_timer(0.1, self.monitor_loop)

        self.get_logger().info('Policy Metrics Recorder Node initialized and waiting for goal...')

    def feedback_callback(self, msg):
        if not self.recording_active or self.mission_start_time is None:
            return
        # Extract simulation time duration from Nav2 feedback
        self.last_navigation_time = float(msg.feedback.navigation_time.sec) + float(msg.feedback.navigation_time.nanosec) * 1e-9

    def scan_callback(self, msg: LaserScan):
        if not self.recording_active or self.mission_start_time is None:
            return

        valid_ranges = [
            r for r in msg.ranges
            if math.isfinite(r) and r > msg.range_min and r < msg.range_max
        ]

        if not valid_ranges:
            return

        min_dist = min(valid_ranges)
        self.min_obstacle_distance = min(self.min_obstacle_distance, min_dist)

        # Debounce collision to count distinct collision events
        if min_dist < self.COLLISION_THRESHOLD:
            if not self.in_collision:
                self.collision_count += 1
                self.in_collision = True
                self.get_logger().warn(
                    f"Collision event registered! Distance: {min_dist:.3f}m < {self.COLLISION_THRESHOLD}m "
                    f"(Total Collisions: {self.collision_count})"
                )
        else:
            self.in_collision = False

        # Debounced Near-miss detection with hysteresis
        if min_dist < self.NEAR_MISS_THRESHOLD and min_dist >= self.COLLISION_THRESHOLD:
            if not self.in_near_miss:
                self.near_miss_count += 1
                self.in_near_miss = True
                self.get_logger().info(
                    f"Near-miss event registered! Distance: {min_dist:.3f}m "
                    f"(Total Near-misses: {self.near_miss_count})"
                )
        elif min_dist > self.NEAR_MISS_THRESHOLD + self.NEAR_MISS_HYSTERESIS:
            self.in_near_miss = False

    def odom_callback(self, msg: Odometry):
        if not self.recording_active or self.mission_start_time is None:
            return

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Accumulate actual path length
        if self.prev_x is not None and self.prev_y is not None:
            dist = math.sqrt((self.x - self.prev_x)**2 + (self.y - self.prev_y)**2)
            self.actual_path_length += dist

        self.prev_x = self.x
        self.prev_y = self.y

        # Progress check based on linear displacement (prevents spin-in-place from resetting stuck timer)
        if self.previous_progress_pose is None:
            self.previous_progress_pose = (self.x, self.y)
            self.last_progress_time = self.last_navigation_time
        else:
            dx = self.x - self.previous_progress_pose[0]
            dy = self.y - self.previous_progress_pose[1]
            dist_since_last_progress = math.sqrt(dx**2 + dy**2)
            if dist_since_last_progress > 0.05:
                self.last_progress_time = self.last_navigation_time
                self.previous_progress_pose = (self.x, self.y)

    def cmd_vel_callback(self, msg: Twist):
        if not self.recording_active or self.mission_start_time is None:
            return

        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        self.linear_velocities.append(linear_vel)
        self.angular_velocities.append(angular_vel)

        if self.prev_linear_vel is not None and self.prev_angular_vel is not None:
            linear_change = abs(linear_vel - self.prev_linear_vel)
            angular_change = abs(angular_vel - self.prev_angular_vel)
            self.velocity_change.append((linear_change, angular_change))

        self.prev_linear_vel = linear_vel
        self.prev_angular_vel = angular_vel

    def plan_callback(self, msg: Path):
        if not msg.poses:
            return

        # Only take the first plan, ignore replans
        if self.planned_path_length > 0.0:
            return

        planned_length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i - 1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i - 1].pose.position.y
            planned_length += math.sqrt(dx**2 + dy**2)

        self.planned_path_length = planned_length
        self.get_logger().info(f"First global plan received: {len(msg.poses)} poses, length = {planned_length:.2f}m")

    def status_callback(self, msg: GoalStatusArray):
        if not msg.status_list or not self.recording_active:
            return

        # If we are not tracking any goal, look for an active one (status 1 = Accepted, 2 = Executing)
        if self.tracked_goal_id is None:
            for status_item in msg.status_list:
                if status_item.status in [1, 2]:
                    self.tracked_goal_id = tuple(status_item.goal_info.goal_id.uuid)
                    self.mission_start_time = self.get_clock().now()
                    self.last_progress_time = 0.0
                    self.last_navigation_time = 0.0
                    self.previous_progress_pose = None
                    self.get_logger().info(f"Tracking Nav2 Goal ID: {self.tracked_goal_id}")
                    break

        if self.tracked_goal_id is not None:
            tracked_item = None
            for status_item in msg.status_list:
                if tuple(status_item.goal_info.goal_id.uuid) == self.tracked_goal_id:
                    tracked_item = status_item
                    break

            if tracked_item is not None:
                status_code = tracked_item.status
                if status_code in [4, 5, 6]:
                    self.get_logger().info(f"Goal status ended with code: {status_code}")
                    if status_code == 4:
                        self.stop_and_save("success")
                    else:
                        self.stop_and_save("aborted")

    def monitor_loop(self):
        if not self.recording_active or self.mission_start_time is None:
            return

        # 1. Stuck timeout check
        stuck_duration = self.last_navigation_time - self.last_progress_time
        if stuck_duration > self.STUCK_TIMEOUT_LIMIT:
            self.get_logger().warn(
                f"Stuck timeout reached! Robot hasn't made progress for {stuck_duration:.1f}s "
                f"(Limit: {self.STUCK_TIMEOUT_LIMIT}s)"
            )
            self.stop_and_save("timeout")
            return

        # 2. Absolute timeout check
        if self.last_navigation_time > self.ABSOLUTE_TIMEOUT_LIMIT:
            self.get_logger().warn(
                f"Absolute simulation timeout reached ({self.last_navigation_time:.1f}s > {self.ABSOLUTE_TIMEOUT_LIMIT}s)"
            )
            self.stop_and_save("timeout")
            return

        # 3. Fallback to ROS elapsed clock time
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.mission_start_time).nanoseconds / 1e9
        if elapsed_time > self.ABSOLUTE_TIMEOUT_LIMIT * 3:
            self.get_logger().warn(f"Mission timeout reached via fallback elapsed clock ({elapsed_time:.1f}s > {self.ABSOLUTE_TIMEOUT_LIMIT * 3}s)")
            self.stop_and_save("timeout")

    def stop_and_save(self, result_type):
        if self.csv_written:
            return
        self.csv_written = True
        self.recording_active = False

        now = self.get_clock().now()

        # Calculate navigation duration
        navigation_time = self.last_navigation_time
        if navigation_time == 0.0 and self.mission_start_time is not None:
            navigation_time = (now - self.mission_start_time).nanoseconds / 1e9

        self.get_logger().info(f"🛑 Terminating recording. Result: {result_type}. Navigation time: {navigation_time:.2f}s")

        # Calculate path efficiency
        path_efficiency = None
        if self.planned_path_length > 0.0:
            path_efficiency = round(self.actual_path_length / self.planned_path_length, 2)

        # Calculate averages and smoothness
        avg_linear_velocity = sum(self.linear_velocities) / len(self.linear_velocities) if self.linear_velocities else 0.0
        avg_angular_velocity = sum(self.angular_velocities) / len(self.angular_velocities) if self.angular_velocities else 0.0

        avg_linear_smoothness = sum(v[0] for v in self.velocity_change) / len(self.velocity_change) if self.velocity_change else 0.0
        avg_angular_smoothness = sum(v[1] for v in self.velocity_change) / len(self.velocity_change) if self.velocity_change else 0.0

        min_obstacle_distance = self.min_obstacle_distance if self.min_obstacle_distance != float('inf') else 0.0

        # Read metadata parameters
        policy_name = self.get_parameter('policy_name').value
        environment = self.get_parameter('environment').value
        scenario = self.get_parameter('scenario').value
        run_number = self.get_parameter('run_number').value
        output_directory = self.get_parameter('output_directory').value

        # Prepare CSV row
        row = {
            'policy': policy_name,
            'environment': environment,
            'scenario': scenario,
            'run': run_number,
            'result': result_type,
            'navigation_time': round(navigation_time, 2),
            'planned_path_length': round(self.planned_path_length, 2),
            'actual_path_length': round(self.actual_path_length, 2),
            'path_efficiency': path_efficiency,
            'collision_count': self.collision_count,
            'near_miss_count': self.near_miss_count,
            'minimum_obstacle_distance': round(min_obstacle_distance, 3),
            'average_linear_velocity': round(avg_linear_velocity, 3),
            'average_angular_velocity': round(avg_angular_velocity, 3),
            'linear_velocity_smoothness': round(avg_linear_smoothness, 3),
            'angular_velocity_smoothness': round(avg_angular_smoothness, 3),
        }

        # Resolve CSV path
        output_dir_expanded = os.path.expanduser(output_directory)
        os.makedirs(output_dir_expanded, exist_ok=True)
        csv_filepath = os.path.join(output_dir_expanded, 'mppi_evaluation_metrics.csv')

        # Write to CSV (append mode)
        file_exists = os.path.exists(csv_filepath)
        write_header = not file_exists or os.path.getsize(csv_filepath) == 0

        try:
            with open(csv_filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
            self.get_logger().info(f"Saved experiment metrics to: {csv_filepath}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to write to CSV file {csv_filepath}: {str(e)}")

        # Shut down the node gracefully
        self.get_logger().info("Gracefully shutting down policy_metrics_node...")
        self.shutdown_timer = self.create_timer(0.5, self.shutdown_callback)

    def shutdown_callback(self):
        self.destroy_node()
        rclpy.try_shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = PolicyMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.csv_written:
            node.stop_and_save("aborted")
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
