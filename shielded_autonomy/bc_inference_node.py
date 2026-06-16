import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatusArray

import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

import numpy as np
import onnxruntime as ort
import os

class BCInferenceNode(Node):
    def __init__(self):
        super().__init__('bc_inference_node')

        # ── Parameters ────────────────────────────────────────────────────
        # Paths to model and normalization stats.
        self.declare_parameter('model_path',
            os.path.expanduser('~/nav2_ws/src/shielded_autonomy/bc_model_1/bc_policy.onnx'))
        self.declare_parameter('norm_stats_path',
            os.path.expanduser('~/nav2_ws/src/shielded_autonomy/bc_model_1/bc_norm_stats.npz'))
        self.declare_parameter('scan_range_max', 3.5)
        self.declare_parameter('inference_hz', 10.0)

        # Safety limits — clip network output to physically safe values.
        # same values as MPPI expert
        self.declare_parameter('max_linear_x',  0.26)
        self.declare_parameter('max_angular_z', 1.9)

        model_path          = self.get_parameter('model_path').value
        norm_stats_path     = self.get_parameter('norm_stats_path').value
        self.scan_range_max = self.get_parameter('scan_range_max').value
        inference_hz        = self.get_parameter('inference_hz').value
        self.max_linear_x   = self.get_parameter('max_linear_x').value
        self.max_angular_z  = self.get_parameter('max_angular_z').value

        # ── Load ONNX model ───────────────────────────────────────────────
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(model_path)

        self.ort_session = ort.InferenceSession(model_path)
        self.input_name  = self.ort_session.get_inputs()[0].name
        self.get_logger().info(f'ONNX model loaded: {model_path}')

        # ── Load normalization statistics ─────────────────────────────────
        if not os.path.exists(norm_stats_path):
            self.get_logger().error(f'Norm stats not found: {norm_stats_path}')
            raise FileNotFoundError(norm_stats_path)

        stats         = np.load(norm_stats_path)
        self.obs_mean = stats['obs_mean'].astype(np.float32)  # shape (41,)
        self.obs_std  = stats['obs_std'].astype(np.float32)   # shape (41,)
        self.act_mean = stats['act_mean'].astype(np.float32)  # shape (2,)
        self.act_std  = stats['act_std'].astype(np.float32)   # shape (2,)
        self.get_logger().info('Normalization statistics loaded.')

        # ── TF2 ───────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Latest state storage ──────────────────────────────────────────
        self.latest_scan     = None
        self.latest_odom_vel = None
        self.current_goal    = None

        # ── Monitor navigation status ─────────────────────────────────────
        self.navigating = False

        # ── QoS for sensor topics ─────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile=sensor_qos)

        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10)

        self.status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback, 10)

        # ── Publisher ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_bc', 10)

        # ── Inference timer ───────────────────────────────────────────────
        timer_period = 1.0 / inference_hz
        self.timer = self.create_timer(timer_period, self.inference_loop)

        self.get_logger().info(
            f'BC Inference Node ready at {inference_hz}Hz. '
            f'Waiting for plan...'
        )

    # ── Callbacks  ─────────────────────────────
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, self.scan_range_max)
        ranges = np.nan_to_num(ranges, nan=self.scan_range_max)
        self.latest_scan = ranges[::10]  # downsmaple 360 → 36

    def odom_callback(self, msg: Odometry):
        self.latest_odom_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ], dtype=np.float32)

    def plan_callback(self, msg: Path):
        # Always update goal
        if not msg.poses:
            return
        self.current_goal = msg.poses[-1]
    
    def status_callback(self, msg: GoalStatusArray):
        if not msg.status_list:
            self.navigating = False
            return
        

        # Check if ANY goal in the entire history array is currently running (1 = Accepted, 2 = Executing)
        is_currently_active = any(status_item.status in [1, 2] for status_item in msg.status_list)

        if is_currently_active:
            self.navigating = True
        else:
            # If nothing is active, but we were just moving, we must have finished/cancelled!
            if self.navigating: 
                self.get_logger().info("Nav2 Goal Canceled, Aborted, or Succeeded. Stopping BC Policy.")
                self.navigating = False
                self.current_goal = None
                
                # Hard stop the robot
                stop = Twist()
                self.cmd_pub.publish(stop)

    # ── Inference loop ────────────────────────────────────────────────────
    def inference_loop(self):
        if not self.navigating:
            return

        # Safety gate — don't predict until all data is available.
        if (self.latest_scan     is None or
            self.latest_odom_vel is None or
            self.current_goal    is None):
            return

        # ── Get goal in robot frame ───────────────────────────────────────
        # Use rclpy.time.Time() for latest available transform
        # to avoid future extrapolation warning.
        goal_local = self.get_goal_in_robot_frame()
        if goal_local is None:
            return

        # ── Build observation vector ──────────────────────────────────────
        # [36 scan rays | 2 odom_vel | 2 goal_local] = 40 dims
        obs = np.concatenate([
            self.latest_scan,
            self.latest_odom_vel,
            goal_local
        ]).astype(np.float32)

        # ── Normalize ─────────────────────────────────────────────────────
        # Apply same normalization as training.
        obs_norm = (obs - self.obs_mean) / self.obs_std

        # ── ONNX inference ────────────────────────────────────────────────
        # Reshape to (1, 40) — ONNX expects a batch dimension.
        input_tensor = obs_norm.reshape(1, -1)
        ort_outputs  = self.ort_session.run(
            None, {self.input_name: input_tensor}
        )

        # ── Denormalize output ────────────────────────────────────────────
        # Reverse normalization to get real cmd_vel values.
        action_norm = ort_outputs[0][0]  # shape (2,)
        action = (action_norm * self.act_std) + self.act_mean

        # ── Safety clipping ───────────────────────────────────────────────
        # Clip to physical limits of TurtleBot3.
        # Protects against out-of-distribution observations causing
        # extreme network outputs that could damage the robot.
        linear_x  = float(np.clip(action[0],
                                  -self.max_linear_x,
                                   self.max_linear_x))
        angular_z = float(np.clip(action[1],
                                  -self.max_angular_z,
                                   self.max_angular_z))

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def get_goal_in_robot_frame(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.current_goal.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            goal_robot_frame = tf2_geometry_msgs.do_transform_pose_stamped(
                self.current_goal, transform
            )

            dx = goal_robot_frame.pose.position.x
            dy = goal_robot_frame.pose.position.y

            goal_distance = np.sqrt(dx**2 + dy**2)
            goal_angle    = np.arctan2(dy, dx)

            # _, _, dtheta = euler_from_quaternion([
            #     goal_robot_frame.pose.orientation.x,
            #     goal_robot_frame.pose.orientation.y,
            #     goal_robot_frame.pose.orientation.z,
            #     goal_robot_frame.pose.orientation.w,
            # ]) , dtheta

            return np.array([goal_distance, goal_angle],
                            dtype=np.float32)

        except TransformException as e:
            self.get_logger().warn(
                f'TF not ready: {e}', throttle_duration_sec=2.0)
            return None


def main(args=None):
    rclpy.init(args=args)
    node = BCInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.get_logger().info('Shutting down — robot stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()