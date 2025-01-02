#!/usr/bin/env python
import rospy
import rosbag
import sys
import os
import csv
from geometry_msgs.msg import TransformStamped

def extract_pose_data(bag_file, output_file):
    bag = rosbag.Bag(bag_file)
    # print(f"Extracting pose data from {bag_file} to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/tool_link_ee_pose']):
            if topic == "/tool_link_ee_pose":
                # if isinstance(msg, TransformStamped):
                writer.writerow({
                    'x': msg.transform.translation.x,
                    'y': msg.transform.translation.y,
                    'z': msg.transform.translation.z,
                    'qx': msg.transform.rotation.x,
                    'qy': msg.transform.rotation.y,
                    'qz': msg.transform.rotation.z,
                    'qw': msg.transform.rotation.w
                })
                rospy.loginfo(f"Pose data extracted: ")
            else:
                rospy.loginfo(f"NO data extracted: {topic}")
    bag.close()
    print(f"Pose data extracted to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_pose_data.py <input.bag> <output.csv>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_pose_data(bag_file, output_file)
