# pallet_detection_and_segmentation

**ROS2 Package for Detecting and Segmenting Pallets in Manufacturing and Warehousing Environments**

This ROS2 package provides object detection and semantic segmentation capabilities for pallets in real-time, designed for mobile robotics applications in manufacturing and warehousing environments. The system enables efficient pallet recognition and segmentation using image and depth data.

## Objectives
- **Pallet Detection:** Detect pallets in various environments using object detection models.
- **Semantic Segmentation:** Segment pallets and ground surfaces using semantic segmentation models.

## Dataset Acquisition and Preparation

### Dataset for Detection
The dataset for pallet detection can be accessed through the following link:
- [Pallet Detection Dataset](https://app.roboflow.com/david-akhihiero-pvxdr/pallet-dezmj/deploy)

### Dataset for Segmentation
The dataset for semantic segmentation (pallets and ground) can be accessed through the following link:
- [Pallet and Ground Segmentation Dataset](https://app.roboflow.com/david-akhihiero-pvxdr/pallet_and_ground/3)

### Data Preparation
1. **Annotation:** The dataset has been annotated using **Roboflow**, a tool for creating high-quality datasets with easy-to-use annotation features for both object detection and segmentation tasks.
2. **Dataset Organization:**
   - **Pallet Detection Dataset:**
     - **Training Set:** 95% of the dataset
     - **Validation Set:** 4% of the dataset
     - **Test Set:** 1% of the dataset
   - **Pallet and Ground Segmentation Dataset:**
     - **Training Set:** 87% of the dataset
     - **Validation Set:** 8% of the dataset
     - **Test Set:** 4% of the dataset
3. **Data Augmentation:** To improve model robustness, I applied the following augmentation techniques:
   - **Crop:** 0% Minimum Zoom, 20% Maximum Zoom
   - **Rotation:** Between -15° and +15°
   - **Brightness:** Between -15% and +15%
   - **Blur:** Up to 2.5px
   - **Cutout:** 3 boxes with 10% size each

## Object Detection and Semantic Segmentation

### Model Development
1. **Object Detection:**
   - **Model Used:** YOLOv11, which is optimized for real-time object detection. I use YOLOv11 to detect pallets in images captured from cameras.
   - **Output:** The object detection model provides bounding boxes for pallets detected in the image.
   - **Performance Metrics:**
     - **mAP:** 62.7%
     - **Precision:** 60.7%
     - **Recall:** 58.3%

2. **Semantic Segmentation:**
   - **Model Used:** SegFormer, a lightweight segmentation model. I use SegFormer to segment both the pallets and ground areas in images, providing pixel-wise classification.
   - **Output:** A segmentation mask indicating the pixel locations of the pallets and ground.

### Training and Fine-Tuning
- I train and fine-tune the models using the annotated dataset to optimize performance in manufacturing and warehousing environments.
- Fine-tuning is done with a focus on robustness under varying environmental conditions, such as lighting changes and occlusions.

### Performance Metrics for Segmentation
- **Test Loss:** 0.3476
- **Test Mean Accuracy:** 91.43%
- **Test Mean IoU:** 49.59%

These metrics reflect the performance of the segmentation model in identifying and segmenting pallets and ground surfaces, with a focus on accuracy and the intersection over union (IoU) of the segmented regions.

### Performance Evaluation
- **Object Detection:** I use **mean Average Precision (mAP)** as the evaluation metric for detecting pallets accurately.
- **Segmentation:** The **Intersection over Union (IoU)** metric is used to evaluate segmentation accuracy for pallets and ground surfaces.
- Models are tested under different lighting, occlusion, and environmental conditions to ensure robustness.

## ROS2 Node Development

### ROS2 Package Structure

This ROS2 package is structured to include nodes for detecting pallets and segmenting images. These nodes are organized as follows:

1. **Node for Object Detection and Segmentation:**
   - This node utilizes the **YOLOv11** model for pallet detection and the **SegFormer** model for semantic segmentation of pallets and ground surfaces.
   
2. **Topics Used:**
   - **Input:**
     - `/robot1/zed2i/left/image_rect_color`: Color image topic for detection and segmentation.
     - `/camera/depth/image_raw`: Depth image topic for enhanced segmentation and depth information.
   
   - **Output:**
     - The detected bounding boxes and segmentation masks are displayed for visualization.

### Launch File

To starts the detection and segmentation process:
```bash
ros2 run pallet_detection_and_segmentation detect_and_segment.py
