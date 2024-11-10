#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.declare_parameter('image_directory', '/home/davidakhihiero/Downloads/pallet.v2i.yolov11/train/images')
        self.declare_parameter('publish_rate', 1.0)  # images per second
        
        self.bridge = CvBridge()
        self.image_directory = self.get_parameter('image_directory').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.image_files = sorted([f for f in os.listdir(self.image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.image_index = 0

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_image)
        self.get_logger().info(f"Publishing images from {self.image_directory} at {self.publish_rate} Hz.")

    def publish_image(self):
        if self.image_index >= len(self.image_files):
            self.get_logger().info("Reached the end of the image directory, restarting...")
            self.image_index = 0
        
        image_path = os.path.join(self.image_directory, self.image_files[self.image_index])
        self.get_logger().info(f"Publishing image {image_path}")
        
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            self.get_logger().warn(f"Failed to read image at {image_path}")
            self.image_index += 1
            return

        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.publisher_.publish(ros_image)
        self.image_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
