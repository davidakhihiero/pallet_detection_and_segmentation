#!/usr/bin/env python3

from setuptools import setup
from glob import glob
import os

package_name = 'pallet_detection_and_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': 'src'}, 
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'torch', 'transformers', 'opencv-python'],  
    zip_safe=True,
    author='David Akhihiero',
    author_email='davidakhihiero@gmail.com',
    description='A ROS2 package for pallet detection and segmentation using YOLO and Segformer.',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_and_segment = pallet_detection_and_segmentation.detect_and_segment:main',
            'image_publisher = pallet_detection_and_segmentation.image_publisher:main',
            'video_publisher = pallet_detection_and_segmentation.video_publisher:main',
        ],
    },
)
