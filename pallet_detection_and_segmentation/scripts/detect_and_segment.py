#!/usr/bin/env python3
# Author : David Akhihiero

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os
import numpy as np
import ament_index_python
from pathlib import Path
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


class DetectAndSegment(Node):
    def __init__(self):
        super().__init__('detect_and_segment')
        self.bridge = CvBridge()

        # Load YOLO v11 for detection
        package_share_directory = ament_index_python.get_package_share_directory('pallet_detection_and_segmentation')

        workspace_root = Path(package_share_directory).parents[3]  

        src_directory = workspace_root / 'src' / 'pallet_detection_and_segmentation'

        detection_model_path = src_directory / 'models' / 'detection_model.pt'

        segmentation_model_path = src_directory / 'models' / 'segmentation_model'
        
        # img2 = src_directory / 'models' / 'image.jpg'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
        self.yolo_model = YOLO(detection_model_path).to(device)

        # Load Segformer for segmentation
        self.segformer_extractor = SegformerFeatureExtractor.from_pretrained(segmentation_model_path)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(segmentation_model_path)
        self.segformer_model.eval()

        # Subscribing to image topics
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.latest_image = None
        self.latest_depth = None


    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.perform_detection_and_segmentation()

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')


    def perform_detection_and_segmentation(self):
        if self.latest_image is not None:
            # Run YOLO for object detection
            results = self.yolo_model(self.latest_image)

            # Check if results contain boxes for detection
            detections = []

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates
                        conf = box.conf[0].item() 
                        cls = int(box.cls[0])
                        if cls == 0:
                            cls = "Pallet" # Class

                        detections.append((x1, y1, x2, y2, conf, cls))
                    

            # Run Segformer for segmentation
            inputs = self.segformer_extractor(images=self.latest_image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)
                # print("OUTPUTS")
                # print(outputs[0].shape[2])
  
            logits = outputs[0]
            #logits = logits - 2*logits.min()
            # print("LOGITS")
            # print(logits.shape[1])
            # print("Logits shape:", logits.shape)  # Should reflect the number of classes
            # print("Sample logits values for a few pixels:", logits[:, :, :5, :5])

            segmentation_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            # print("Classes in segmentation mask:", np.unique(segmentation_mask))
            # print(segmentation_mask)

            segmentation_mask = self.apply_color_map(segmentation_mask)

            # Visualize detection and segmentation results
            self.display_results(detections, segmentation_mask)

    def apply_color_map(self, segmentation_mask):
    
        color_map = {
            0: [0, 0, 0],      # Background: None
            1: [0, 0, 255],        # Ground: Red
            2: [0, 255, 0],      # Pallet: Green
        }

       
        color_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)

        for class_label, color in color_map.items():
            color_mask[segmentation_mask == class_label] = color

        return color_mask

    def display_results(self, detections, segmentation_mask):
        img = np.array(self.latest_image) if not isinstance(self.latest_image, np.ndarray) else self.latest_image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
     
        segmentation_mask = segmentation_mask.squeeze() 

        mask_resized = cv2.resize(segmentation_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        

        # print("MASK SIZE")
        # print(segmentation_mask.shape)
        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{cls} {conf*100:.0f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    
        overlay = cv2.addWeighted(img, 0.7, mask_resized, 0.3, 0)

        cv2.imshow('Detection and Segmentation', overlay)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectAndSegment()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
