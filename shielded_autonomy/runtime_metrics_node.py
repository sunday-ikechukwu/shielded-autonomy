#!/usr/bin/env python3
"""
runtime_metrics_node.py

ROS 2 Humble Python node to evaluate the performance of the runtime shared-autonomy
framework (safety monitor, arbitration, and expert interventions).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
import math
import csv
import os

class RuntimeMetricsNode(Node):
    def __init__(self):
        super().__init__('runtime_metrics_node')

        # --- Parameters ---
        self.declare_parameter('policy_name', 'shared_autonomy')
        self.declare_parameter('environment', 'unknown_environment')
        self.declare_parameter('scenario', 'scenario_1')
        self.declare_parameter('run_number', 1)
        self.declare_parameter('output_directory', '~/nav2_ws/src/shielded_autonomy/results/runtime_shared_autonomy')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)# Ensure simulation clock is used by default

        # --- Constants ---
        self.COLLISION_THRESHOLD = 0.24  # Reused from metrics_recorder.py
        self.NEAR_MISS_THRESHOLD = 0.35  # Reused from metrics_recorder.py (local_costmap inflation radius)
        self.NEAR_MISS_HYSTERESIS = 0.05 # 5cm hysteresis to prevent near-miss chattering
        
        # Stuck and Timeout limits (in simulation seconds)
        self.STUCK_TIMEOUT_LIMIT = 50.0   
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

        self.total_actual_path_length = 0.0
        self.distance_under_expert_control = 0.0

        # Timing (Master Timeline sourced from Nav2 feedback)
        self.mission_start_time = None
        self.last_navigation_time = 0.0
        self.last_progress_time = 0.0
        self.previous_progress_pose = None

        # Shared-Autonomy/Intervention metrics
        self.active_mode = 'BC_POLICY'
        self.latest_raw_reason = ""
        
        self.intervention_count = 0
        self.stable_recovery_count = 0
        self.total_expert_control_time = 0.0
        self.maximum_intervention_duration = 0.0
        self.intervention_start_nav_time = None
        self.collision_in_current_intervention = False

        # Stability Window for Recoveries (using Nav2 Master Timeline)
        self.pending_recoveries = []  

        # Takeover obstacle distance tracking
        self.current_min_scan_dist = float('inf')
        self.takeover_distances = []

        # Safety tracking
        self.min_obstacle_distance = float('inf')
        self.collision_count = 0
        self.near_miss_count = 0

        # --- QoS for sensor topics ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=sensor_qos)
        
        # Subscribe to existing runtime topics already produced by the framework
        self.cmd_vel_bc_sub = self.create_subscription(Twist, '/cmd_vel_bc', self.cmd_vel_bc_callback, 10)
        self.cmd_vel_mppi_sub = self.create_subscription(Twist, '/cmd_vel_mppi', self.cmd_vel_mppi_callback, 10)
        self.cmd_vel_selected_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_selected_callback, 10)
        
        self.mode_sub = self.create_subscription(String, '/active_mode', self.mode_callback, 10)
        self.reason_sub = self.create_subscription(String, '/trigger_reason', self.reason_callback, 10)
        
        # Action status and feedback
        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)
        self.feedback_sub = self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )

        # --- Periodic loop to monitor timeout & stability windows ---
        self.timer = self.create_timer(0.1, self.monitor_loop)

        self.get_logger().info('Runtime Metrics Recorder Node initialized and waiting for goal...')

    def feedback_callback(self, msg):
        if not self.recording_active or self.mission_start_time is None:
            return
        # Extract simulation time duration from Nav2 feedback
        self.last_navigation_time = float(msg.feedback.navigation_time.sec) + float(msg.feedback.navigation_time.nanosec) * 1e-9

    def reason_callback(self, msg: String):
        if not self.recording_active or self.mission_start_time is None:
            return
        self.latest_raw_reason = msg.data

    def mode_callback(self, msg: String):
        if not self.recording_active or self.mission_start_time is None:
            return

        new_mode = msg.data
        if new_mode == self.active_mode:
            return

        # Transition INTO EXPERT (Intervention Start)
        if new_mode == 'EXPERT' and self.active_mode != 'EXPERT':
            self.intervention_start_nav_time = self.last_navigation_time
            self.collision_in_current_intervention = False
            self.intervention_count += 1
            
            # Record current minimum scan distance at takeover
            dist_at_takeover = self.current_min_scan_dist
            if dist_at_takeover != float('inf'):
                self.takeover_distances.append(dist_at_takeover)

            # Invalidate any pending recoveries in their stability window
            for pending in self.pending_recoveries:
                pending['has_failed'] = True
            self.pending_recoveries.clear()
            
            self.get_logger().warn(
                f"Takeover [{self.intervention_count}]: MPPI Expert took control! "
                f"Takeover obstacle distance: {dist_at_takeover:.3f}m"
            )

        # Transition OUT of EXPERT (Intervention End)
        elif self.active_mode == 'EXPERT' and new_mode != 'EXPERT':
            if self.intervention_start_nav_time is not None:
                duration = self.last_navigation_time - self.intervention_start_nav_time
                self.total_expert_control_time += duration
                self.maximum_intervention_duration = max(self.maximum_intervention_duration, duration)

                self.get_logger().info(f"Control returned to BC. Intervention duration: {duration:.2f}s")

                # If no collision occurred during this intervention, start its stability window
                if not self.collision_in_current_intervention:
                    self.pending_recoveries.append({
                        'end_time': self.last_navigation_time,
                        'has_failed': False
                    })
                    self.get_logger().info("Starting 2-second stability window for this recovery...")
                else:
                    self.get_logger().warn("Collision occurred during expert takeover. Ineligible for recovery count.")

                self.intervention_start_nav_time = None

        self.active_mode = new_mode

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
        self.current_min_scan_dist = min_dist

        # Debounce collision
        if min_dist < self.COLLISION_THRESHOLD:
            if not self.in_collision:
                self.in_collision = True
                if self.active_mode == 'EXPERT':
                    self.collision_in_current_intervention = True
                
                # Invalidate all active pending recoveries in stability window
                for pending in self.pending_recoveries:
                    pending['has_failed'] = True
                self.pending_recoveries.clear()
                
                self.get_logger().warn(f"Collision event registered! Distance: {min_dist:.3f}m")
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
            self.total_actual_path_length += dist

            # Accumulate distance under expert control
            if self.active_mode == 'EXPERT':
                self.distance_under_expert_control += dist

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

    # Subscriptions required by parameter/framework but not used directly in metrics computation:
    def cmd_vel_bc_callback(self, msg: Twist):
        pass

    def cmd_vel_mppi_callback(self, msg: Twist):
        pass

    def cmd_vel_selected_callback(self, msg: Twist):
        pass

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

        # Check pending recoveries stability window (2.0s) against master Nav2 timeline
        active_pending = []
        for pending in self.pending_recoveries:
            if pending['has_failed']:
                continue
            elapsed_stability = self.last_navigation_time - pending['end_time']
            if elapsed_stability >= 2.0:
                self.stable_recovery_count += 1
                self.get_logger().info(f"Stability window passed! Stable recovery: {self.stable_recovery_count}")
            else:
                active_pending.append(pending)
        self.pending_recoveries = active_pending

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
        now = self.get_clock().now()
        elapsed_time = (now - self.mission_start_time).nanoseconds / 1e9
        if elapsed_time > self.ABSOLUTE_TIMEOUT_LIMIT * 3:
            self.get_logger().warn(f"Mission timeout reached via fallback elapsed clock ({elapsed_time:.1f}s > {self.ABSOLUTE_TIMEOUT_LIMIT * 3}s)")
            self.stop_and_save("timeout")

    def stop_and_save(self, result_type):
        if self.csv_written:
            return
        self.csv_written = True
        self.recording_active = False

        # Handle active intervention at termination using master Nav2 timeline
        if self.active_mode == 'EXPERT' and self.intervention_start_nav_time is not None:
            duration = self.last_navigation_time - self.intervention_start_nav_time
            self.total_expert_control_time += duration
            self.maximum_intervention_duration = max(self.maximum_intervention_duration, duration)
            self.intervention_start_nav_time = None

        # Evaluate any remaining pending recoveries (they successfully got the robot to the goal/termination without failure)
        for pending in self.pending_recoveries:
            if not pending['has_failed']:
                self.stable_recovery_count += 1
        self.pending_recoveries.clear()

        # Measure navigation duration directly from Nav2 action feedback
        navigation_time = self.last_navigation_time
        # Fallback to manual node clock calculation ONLY if feedback message is missing (0.0)
        if navigation_time == 0.0 and self.mission_start_time is not None:
            now = self.get_clock().now()
            navigation_time = (now - self.mission_start_time).nanoseconds / 1e9

        # Averages & Percentages (No min clamping to expose timing issues/bugs)
        avg_intervention_duration = 0.0
        if self.intervention_count > 0:
            avg_intervention_duration = self.total_expert_control_time / self.intervention_count

        expert_time_percentage = 0.0
        if navigation_time > 0.0:
            expert_time_percentage = (self.total_expert_control_time / navigation_time) * 100.0

        expert_distance_percentage = 0.0
        if self.total_actual_path_length > 0.0:
            expert_distance_percentage = (self.distance_under_expert_control / self.total_actual_path_length) * 100.0

        # Autonomy Metrics
        autonomy_time_percentage = 100.0 - expert_time_percentage
        autonomy_distance_percentage = 100.0 - expert_distance_percentage

        # Mean Time Between Interventions (MTBI)
        mean_time_between_interventions = 0.0
        if self.intervention_count > 0:
            mean_time_between_interventions = (navigation_time - self.total_expert_control_time) / self.intervention_count
        else:
            mean_time_between_interventions = navigation_time

        # Log a warning if expert control time exceeds navigation time
        if self.total_expert_control_time > navigation_time:
            self.get_logger().warn(
                f"Expert control time ({self.total_expert_control_time:.2f}s) "
                f"exceeds navigation time ({navigation_time:.2f}s)."
            )

        # Takeover obstacle distance metrics
        average_takeover_obstacle_distance = sum(self.takeover_distances) / len(self.takeover_distances) if self.takeover_distances else 0.0
        minimum_takeover_obstacle_distance = min(self.takeover_distances) if self.takeover_distances else 0.0
        maximum_takeover_obstacle_distance = max(self.takeover_distances) if self.takeover_distances else 0.0

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
            'intervention_count': self.intervention_count,
            'stable_recovery_count': self.stable_recovery_count,
            'navigation_time': round(navigation_time, 2),
            'expert_control_time': round(self.total_expert_control_time, 2),
            'average_intervention_duration': round(avg_intervention_duration, 2),
            'maximum_intervention_duration': round(self.maximum_intervention_duration, 2),
            'distance_under_expert_control': round(self.distance_under_expert_control, 2),
            'expert_distance_percentage': round(expert_distance_percentage, 2),
            'expert_time_percentage': round(expert_time_percentage, 2),
            'autonomy_time_percentage': round(autonomy_time_percentage, 2),
            'autonomy_distance_percentage': round(autonomy_distance_percentage, 2),
            'mean_time_between_interventions': round(mean_time_between_interventions, 2),
            'average_takeover_obstacle_distance': round(average_takeover_obstacle_distance, 3),
            'minimum_takeover_obstacle_distance': round(minimum_takeover_obstacle_distance, 3),
            'maximum_takeover_obstacle_distance': round(maximum_takeover_obstacle_distance, 3),
        }

        # Resolve CSV path
        output_dir_expanded = os.path.expanduser(output_directory)
        os.makedirs(output_dir_expanded, exist_ok=True)
        csv_filepath = os.path.join(output_dir_expanded, 'unknown_environent_runtime_framework_metrics_2.csv')

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
            self.get_logger().info(f"Saved runtime metrics to: {csv_filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to write to CSV file {csv_filepath}: {str(e)}")

        # Shut down the node gracefully
        self.get_logger().info("Gracefully shutting down runtime_metrics_node...")
        self.shutdown_timer = self.create_timer(0.5, self.shutdown_callback)

    def shutdown_callback(self):
        self.destroy_node()
        rclpy.try_shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RuntimeMetricsNode()
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
