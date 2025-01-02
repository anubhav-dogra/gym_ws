#!/usr/bin/env python

import rosbag
import sys
import os
from cv_bridge import CvBridge
import cv2

def extract_images(bag_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bag = rosbag.Bag(bag_file)
    bridge = CvBridge()
    count = 1
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = t.to_nsec()
            image_filename = os.path.join(output_dir, f"image_{count}.png")
            cv2.imwrite(image_filename, cv_image)
        except Exception as e:
            print(f"Failed to process image: {e}")

        count += 1
    
    bag.close()
    print(f"Images extracted to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_images.py <input.bag> <output_dir>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_dir = sys.argv[2]
    extract_images(bag_file, output_dir)
