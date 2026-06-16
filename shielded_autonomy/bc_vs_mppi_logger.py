"""
bc_vs_mppi_logger.py
Subscribes to both /cmd_vel_bc and /cmd_vel simultaneously.
Logs synchronized pairs with timestamp for proper comparison.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import os
from datetime import datetime

class BCvsMPPILogger(Node):
    def __init__(self):
        super().__init__('bc_vs_mppi_logger')

        self.latest_bc   = None
        self.latest_mppi = None

        self.bc_sub = self.create_subscription(
            Twist, '/cmd_vel_bc', self.bc_callback, 10)
        self.mppi_sub = self.create_subscription(
            Twist, '/cmd_vel', self.mppi_callback, 10)

        # Log to CSV for analysis
        log_path = os.path.expanduser(
            '~/nav2_ws/src/shielded_autonomy/bc_vs_mppi_log.csv'
        )
        self.csv_file = open(log_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'timestamp',
            'bc_linear_x', 'bc_angular_z',
            'mppi_linear_x', 'mppi_angular_z',
            'linear_error', 'angular_error'
        ])

        self.create_timer(0.1, self.log_callback)
        self.get_logger().info('BC vs MPPI logger ready.')

    def bc_callback(self, msg: Twist):
        self.latest_bc = msg

    def mppi_callback(self, msg: Twist):
        self.latest_mppi = msg

    def log_callback(self):
        if self.latest_bc is None or self.latest_mppi is None:
            return

        bc_lx  = self.latest_bc.linear.x
        bc_wz  = self.latest_bc.angular.z
        mp_lx  = self.latest_mppi.linear.x
        mp_wz  = self.latest_mppi.angular.z

        self.writer.writerow([
            self.get_clock().now().nanoseconds,
            bc_lx, bc_wz,
            mp_lx, mp_wz,
            abs(bc_lx - mp_lx),
            abs(bc_wz - mp_wz)
        ])

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BCvsMPPILogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()