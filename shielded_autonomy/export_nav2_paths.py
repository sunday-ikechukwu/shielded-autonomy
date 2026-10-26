import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np
import os
import threading

class AutoFirstPathExporter(Node):
    def __init__(self):
        super().__init__('nav_auto_path_exporter')

        self.output_dir = os.path.expanduser('~/mujoco_sim/models/robotis_mujoco_menagerie/robotis_tb3/eval_paths')
        os.makedirs(self.output_dir, exist_ok=True)

        self.current_scenario = 1
        self.armed = False
        self.plan_lock = threading.Lock()
        
        # Event to signal the main thread that a path was captured
        self.capture_event = threading.Event()

        # Subscribe to /plan
        self.plan_sub = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10
        )

    def plan_callback(self, msg: Path):
        with self.plan_lock:
            # ONLY trigger if the script is armed AND a path actually exists
            if self.armed and msg.poses:
                
                # 1. Extract and format the waypoints
                waypoints = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
                plan_array = np.array(waypoints, dtype=np.float32)

                # 2. Save immediately to the .npy file
                file_name = f"scenario_{self.current_scenario}_plan.npy"
                file_path = os.path.join(self.output_dir, file_name)
                np.save(file_path, plan_array)

                self.get_logger().info(
                    f"\n[SUCCESS] Captured initial full path for Scenario {self.current_scenario} "
                    f"({len(plan_array)} points) -> {file_name}"
                )
                
                # 3. DISARM instantly! All future replanning messages will be ignored.
                self.armed = False
                
                # 4. Tell the main thread to move on
                self.capture_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = AutoFirstPathExporter()

    # Spin ROS 2 callbacks in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n=======================================================")
    print("      Nav2 Auto-Trap Path Exporter (First-Path Only)   ")
    print("=======================================================")
    print(f"Saving to: {node.output_dir}\n")

    for scenario_id in range(1, 9):
        node.current_scenario = scenario_id
        
        # 1. Wait for user to get the robot into position
        input(f"--> Place robot at START pose for Scenario {scenario_id}. Press [ENTER] to ARM the script...")
        
        # 2. Arm the script
        with node.plan_lock:
            node.armed = True
            node.capture_event.clear()
            
        print(f"[*] ARMED for Scenario {scenario_id}. Now, send the 2D Nav Goal in RViz!")
        
        # 3. Wait blockingly until the callback captures the first path and sets the event
        node.capture_event.wait()

    print("\nAll 8 scenarios successfully recorded with perfect first-path accuracy! Shutting down.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()