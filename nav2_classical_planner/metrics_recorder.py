# metrics_recorder.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import csv


class MetricsRecorder(Node):
    def __init__(self):
        super().__init__('metrics_recorder')

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(String, '/metrics_cmd', self.cmd_callback, 10)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.nav_status_callback, 10)
        self.create_subscription(NavigateToPose.Impl.FeedbackMessage, '/navigate_to_pose/_action/feedback', self.feedback_callback, 10)

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # --- state ---
        self.goal_logged = False
        self.motion_active = False
        self.motion_end_time = None

        self.goal_x = None
        self.goal_y = None

        self.still_count = 0
        self.STILL_THRESHOLD = 5
        self.VEL_THRESHOLD = 0.02
        self.POS_THRESHOLD = 0.25

        self.x = 0.0
        self.y = 0.0
        self.v = 0.0

        self.path_length = 0.0
        self.planned_path_length = 0.0
        self.prev_x = None
        self.prev_y = None

        self.last_navigation_time = 0
        self.last_recovery_count = 0

        #velocity tracking
        self.linear_velocities = []
        self.angular_velocities = []

        #smoothness tracking
        self.prev_linear_vel = None
        self.prev_angular_vel = None
        self.velocity_change = []

        #Obstacle proximity tracking
        self.min_obstacle_distance = float('inf')
        self.collision_count = 0
        self.near_miss_count = 0
        self.COLLISION_THRESHOLD = 0.24
        self.NEAR_MISS_THRESHOLD = 0.35 # local_costmap inflation radius value

        self.scenario_name = ''
        self.pending_result = None
        self.results = []

        # Track run per scenario to track number of trials for each scenario
        self.scenario_runs = {}
        self.run_count = 0 

        #Specify planner name for later comparison with RL planner
        self.planner_name = "DWB"  # change to RL later

        self.get_logger().info('Metrics recorder ready')

    # ---------------- RESET ----------------
    def reset_metrics(self):
        self.path_length = 0.0
        self.planned_path_length = 0.0
        self.still_count = 0
        self.motion_active = False
        self.motion_end_time = None
        self.goal_logged = False
        self.prev_x = None
        self.prev_y = None
        self.last_navigation_time = 0
        self.last_recovery_count = 0
        self.pending_result = None

        self.linear_velocities.clear()
        self.angular_velocities.clear()
        self.velocity_change.clear()
        self.prev_linear_vel = None
        self.prev_angular_vel = None

        self.min_obstacle_distance = float('inf')
        self.collision_count = 0
        self.near_miss_count = 0 

    # ---------------- FEEDBACK ----------------
    def feedback_callback(self, msg):
        # save last values before they reset to 0 on goal completion
        self.last_navigation_time = msg.feedback.navigation_time.sec
        self.last_recovery_count = msg.feedback.number_of_recoveries
    
    def cmd_vel_callback(self,msg):
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z

        if not self.motion_active: # ignore idle periods
            return

        #store velocities
        self.linear_velocities.append(self.linear_vel)
        self.angular_velocities.append(self.angular_vel)

        # Compute smoothness (change in velocity)
        if self.prev_linear_vel is not None and self.prev_angular_vel is not None:
            linear_change = abs(self.linear_vel - self.prev_linear_vel)
            angular_change = abs(self.angular_vel - self.prev_angular_vel)
            self.velocity_change.append((linear_change, angular_change))
        
        self.prev_linear_vel = self.linear_vel
        self.prev_angular_vel = self.angular_vel

    def scan_callback(self, msg):
        # filter range to discard useless data
        valid_ranges= [
            r for r in msg.ranges
            if math.isfinite(r) and r > msg.range_min and r < msg.range_max
        ]

        # exit if not ray hit an obstacle
        if not valid_ranges:
            return
        
        # get minimium range
        min_dist = min(valid_ranges)
        self.min_obstacle_distance = min(self.min_obstacle_distance, min_dist)

        if min_dist < self.COLLISION_THRESHOLD:
            self.collision_count += 1
        elif min_dist < self.NEAR_MISS_THRESHOLD:
            self.near_miss_count += 1

    # ---------------- CMD CONTROL ----------------
    def cmd_callback(self, msg):
        data = msg.data

        if data.startswith('start:'):
            self.scenario_name = data.split('start:')[1]

             # increment run per scenario
            if self.scenario_name not in self.scenario_runs:
                self.scenario_runs[self.scenario_name] = 0
            self.scenario_runs[self.scenario_name] += 1
            self.run_count = self.scenario_runs[self.scenario_name]

            self.reset_metrics()
            self.get_logger().info(f'Recording started: {self.scenario_name}')
        elif data == 'save':
            self.save_to_csv()

    # ---------------- NAV STATUS ----------------
    def nav_status_callback(self, msg):
        if self.goal_logged:
            return
        
        if not msg.status_list:
            return
        
        status = msg.status_list[-1] # get the latest status
        status_code = status.status

        if status_code not in [4, 5, 6]:
            return
        
        if self.scenario_name == '':
            self.get_logger().warn("Scenario not set. Ignoring result.")
            return
        
        if status.status == 4:
            result_type = "success"
        elif status_code == 5:
            result_type = "failure:canceled"
        elif status_code == 6:
            result_type = "failure:aborted"

        if self.motion_end_time is not None:
            self.log_result(result_type)
        else:
            self.get_logger().warn("Motion not finished yet, delaying result logging")
            self.pending_result = result_type
        

    def log_result(self, result_type):
        ratio = round(self.path_length / self.planned_path_length, 2) if self.planned_path_length > 0 else None

        #__________________Averages____________________________
        avg_linear_velocity = sum(self.linear_velocities) / len(self.linear_velocities) if self.linear_velocities else 0.0
        avg_angular_velocity = sum(self.angular_velocities) / len(self.angular_velocities) if self.angular_velocities else 0.0

        #___________________Smoothness____________________
        avg_linear_smoothness = sum(abs(v[0]) for v in self.velocity_change) / len(self.velocity_change) if self.velocity_change else 0.0
        avg_angular_smoothness = sum(abs(v[1]) for v in self.velocity_change) / len(self.velocity_change) if self.velocity_change else 0.0

        # Min obstacle distance
        min_obstacle_distance = self.min_obstacle_distance if self.min_obstacle_distance != float('inf') else 0.0

        collision_count = self.collision_count
        near_miss_count = self.near_miss_count


        result = {
            'planner': self.planner_name,
            'scenario': self.scenario_name,
            'time_secs': self.last_navigation_time,
            'planned_path_length_m': round(self.planned_path_length, 2),
            'path_length_m': round(self.path_length, 2),
            'path_efficiency': ratio,
            'recovery_count': self.last_recovery_count,
            'result': result_type,
            'run': self.run_count,
            'avg_linear_velocity': round(avg_linear_velocity, 3),
            'avg_angular_velocity': round(avg_angular_velocity, 3),
            'avg_linear_smoothness': round(avg_linear_smoothness, 3),
            'avg_angular_smoothness': round(avg_angular_smoothness, 3),
            'min_obstacle_distance': round(min_obstacle_distance, 3),
            'collision_count': collision_count,
            'near_miss_count': near_miss_count
        }

        self.results.append(result)
        self.goal_logged = True
        self.pending_result = None

        self.save_to_csv()
        self.get_logger().info(f"Result logged: {result}")


    # ---------------- ODOM ----------------
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v = math.sqrt(vx**2 + vy**2)

        # start motion detection
        if not self.motion_active and self.v > self.VEL_THRESHOLD:
            self.motion_active = True
            # self.goal_logged = False
            self.get_logger().info("Motion started")

        # accumulate driven distance
        if self.motion_active and self.prev_x is not None:
            dist = math.sqrt(
                (self.x - self.prev_x)**2 +
                (self.y - self.prev_y)**2
            )
            self.path_length += dist

        # stillness / motion end check
        if self.motion_active:
            if self.goal_x is None:
                self.prev_x = self.x
                self.prev_y = self.y
                return

            dist_to_goal = math.sqrt(
                (self.x - self.goal_x)**2 +
                (self.y - self.goal_y)**2
            )

            if self.v < self.VEL_THRESHOLD:
                self.still_count += 1
            else:
                self.still_count = 0

            if self.still_count >= self.STILL_THRESHOLD:
                self.motion_active = False
                self.motion_end_time = self.get_clock().now()
                self.get_logger().info("Motion ended")

                #If result came early → log it now
                if self.pending_result is not None and not self.goal_logged:
                    self.log_result(self.pending_result)

        self.prev_x = self.x
        self.prev_y = self.y

    # ---------------- PLAN ----------------
    def plan_callback(self, msg):
        if not msg.poses:
            return
        
        # only take the first plan, ignore replans
        if self.planned_path_length > 0:
            return

        # compute planned path length
        planned_length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i - 1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i - 1].pose.position.y
            planned_length += math.sqrt(dx**2 + dy**2)
        self.planned_path_length = planned_length

        # extract goal coordinates
        goal_pose = msg.poses[-1].pose.position
        self.goal_x = goal_pose.x
        self.goal_y = goal_pose.y

        self.get_logger().info(f'Plan received: {len(msg.poses)} poses, length={planned_length:.2f}m')

    # ---------------- SAVE ----------------
    def save_to_csv(self):
        filename = 'nav2_metrics.csv'

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'planner',
                'scenario',
                'run',
                'result',
                'time_secs',
                'planned_path_length_m',
                'path_length_m',
                'path_efficiency',
                'recovery_count',
                'avg_linear_velocity',
                'avg_angular_velocity',
                'avg_linear_smoothness',
                'avg_angular_smoothness',
                'min_obstacle_distance',
                'collision_count',
                'near_miss_count'
            ])
            writer.writeheader()
            writer.writerows(self.results)

        self.get_logger().info(f'Saved to {filename}')


def main():
    rclpy.init()
    node = MetricsRecorder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()