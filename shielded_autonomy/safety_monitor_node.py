import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg    import LaserScan
from std_msgs.msg       import Bool, String  # Added String for reason_pub
from geometry_msgs.msg  import Twist

import numpy as np 

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')

        #___________Parameter_________________________
        self.declare_parameter('lidar_panic_threshold', 0.22)
        self.declare_parameter('lidar_warning_threshold', 0.35)
        self.declare_parameter('action_delta_threshold', 0.8) #0.6
        self.declare_parameter('min_turn_to_compare', 0.2) # 0.15 Sign mismatch limit
        self.declare_parameter('check_hz', 10.0)

        self.lidar_threshold = self.get_parameter('lidar_panic_threshold').value
        self.warning_threshold = self.get_parameter('lidar_warning_threshold').value
        self.delta_threshold = self.get_parameter('action_delta_threshold').value
        self.min_turn_to_compare = self.get_parameter('min_turn_to_compare').value
        check_hz             = self.get_parameter('check_hz').value

        #___________State storages________________________
        self.latest_scan    = None
        self.latest_bc_w    = None     # BC angular velocity
        self.latest_mppi_w  = None     # MPPI angular velocity

        #____________Qos______________________________
        sensor_qos      = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.VOLATILE,
            depth       = 5
        )

        #____________Subscribers____________________________
        self.scan_sub  = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=sensor_qos)
        
        # Listen to BOTH brains!
        self.bc_sub   = self.create_subscription(Twist, '/cmd_vel_bc', self.bc_callback, 10)
        self.mppi_sub = self.create_subscription(Twist, '/cmd_vel_mppi', self.mppi_callback, 10)

        #___________Publishers_________________________________
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)
        self.reason_pub = self.create_publisher(String, '/trigger_reason', 10) # PHASE 2: Reason publisher

        #___________Evaluation Timer_________________________
        self.timer = self.create_timer(1.0 / check_hz, self.safety_check_loop)

        self.get_logger().info(f'Safety Monitor Active. Disagreement Limit: {self.delta_threshold} rad/s, LiDAR Limit: {self.lidar_threshold}m')

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]

        if len(valid_ranges) > 0:
            self.latest_scan = np.min(valid_ranges) 
        else:
            self.latest_scan = float('inf')

    def bc_callback(self, msg: Twist):
        self.latest_bc_w = msg.angular.z

    def mppi_callback(self, msg: Twist):
        self.latest_mppi_w = msg.angular.z

    def safety_check_loop(self):
        if self.latest_scan is None or self.latest_bc_w is None or self.latest_mppi_w is None:
            return
        
        risk_detected = False
        trigger_reason = ""

        # 1. CHECK LIDAR PANIC
        if self.latest_scan < self.lidar_threshold:
            risk_detected = True
            trigger_reason = f"LIDAR PANIC ({self.latest_scan:.2f}m < {self.lidar_threshold}m)"

        # 2. CHECK EARLY WARNING (Inflation Breach)
        elif self.latest_scan < self.warning_threshold:
            risk_detected = True
            trigger_reason = f"INFLATION BREACH ({self.latest_scan:.2f}m < {self.warning_threshold}m)"

        # 3. CHECK BRAIN DISAGREEMENT (Action Delta & Sign Mismatch)
        else:
            delta_w = abs(self.latest_mppi_w - self.latest_bc_w)
            
            # Evaluate topological disagreement
            opposite_sign = (
                np.sign(self.latest_mppi_w) != np.sign(self.latest_bc_w)
                and abs(self.latest_mppi_w) > self.min_turn_to_compare
                and abs(self.latest_bc_w) > self.min_turn_to_compare
            )

            if delta_w > self.delta_threshold:
                risk_detected = True
                trigger_reason = f"POLICY DISAGREEMENT (Delta: {delta_w:.2f} > {self.delta_threshold})"
            elif opposite_sign:
                risk_detected = True
                trigger_reason = f"TURN SIGN MISMATCH (BC={self.latest_bc_w:.2f}, MPPI={self.latest_mppi_w:.2f})"

        # --- PUBLISH STATUS ---
        msg = Bool()
        msg.data = risk_detected
        self.safety_pub.publish(msg)

        # --- PUBLISH REASON AND LOG ---
        if risk_detected:
            reason_msg = String()
            reason_msg.data = trigger_reason
            self.reason_pub.publish(reason_msg)
            self.get_logger().warn(f"SHIELD TRIGGERED: {trigger_reason}", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()