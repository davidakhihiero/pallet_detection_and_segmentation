#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.declare_parameter('video_file', '/home/davidakhihiero/Downloads/videoplayback.mp4')
        self.declare_parameter('publish_rate', 60.0)  # frames per second
        
        self.bridge = CvBridge()
        self.video_file = self.get_parameter('video_file').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file {self.video_file}")
            rclpy.shutdown()
            return
        
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_frame)
        self.get_logger().info(f"Publishing video frames from {self.video_file} at {self.publish_rate} Hz.")
    
    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video file reached, restarting...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            return

        self.get_logger().info("Publishing frame")
        
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(ros_image)

    def destroy(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy()

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
