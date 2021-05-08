#!/usr/bin/env python

from __future__ import print_function

from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from sensor_msgs.msg import Image
import argparse
import cv2
import numpy as np
import rospy


class PublisherNode:

    def __init__(self, filepath=None, is_video=False, topic='/camera/image'):
        self.filepath = filepath
        self.is_video = is_video
        if self.filepath is not None:
            if is_video:
                self.source = cv2.VideoCapture(self.filepath)
                assert self.source.isOpened()
            else:
                self.source = cv2.imread(self.filepath)
                assert self.source is not None
        else:
            self.source = None

        self.image = None
        self.image_pub = rospy.Publisher(topic, Image)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(10)

        self.read_image()

    def read_image(self):
        if self.filepath is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (10, 500)
            font_scale = 1
            font_color = (255, 255, 255)
            line_type = 2
            text = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.image = np.random.randint(255,
                                           size=(720, 1280, 3),
                                           dtype=np.uint8)
            cv2.putText(self.image, text, bottom_left_corner_of_text, font,
                        font_scale, font_color, line_type)
        elif not self.is_video:
            self.image = self.source
        else:
            ret = False
            while not ret:
                ret, self.image = self.source.read()
                if not ret:
                    self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print('shift back to the begining')

    def publish(self):
        try:
            self.read_image()
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(
                self.image, 'bgr8'))
        except CvBridgeError as e:
            print(e)

    def start(self):
        while not rospy.is_shutdown():
            # rospy.loginfo('publishing image')
            if self.image is not None:
                self.publish()
            self.loop_rate.sleep()


def main():
    # parse args
    parser = argparse.ArgumentParser(
        description='Publish image to ROS topic from local file')
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('--is_video', action='store_true')
    opt = parser.parse_args()

    # start ROS node
    rospy.init_node('image_publisher', anonymous=True)
    node = PublisherNode(filepath=opt.input, is_video=opt.is_video)
    try:
        node.start()
    except KeyboardInterrupt:
        print('Shutting down')


if __name__ == '__main__':
    main()