#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from YolactEdgeEngine import YolactEdgeEngine

class SegmentationWorker:

    def __init__(self):
        self.mPubTopic = "/segmentation_image01"
        self.mPub = rospy.Publisher(self.mPubTopic, Image, queue_size=1)
        self.bridge = CvBridge()
        self.engine = YolactEdgeEngine()
        #self.engine.setup()

    def publish(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            cv_img = cv2.imread("/home/ht/Documents/demo.jpg")
            cv_img_out = self.engine.detect(cv_img)
            try:
                ros_imgmsg = self.bridge.cv2_to_imgmsg(cv_img_out, "bgr8")
            except CvBridgeError as e:
                print(e)
            #cv2.imshow("Image", cv_img)
            #cv2.waitKey()

            self.mPub.publish(ros_imgmsg)
            print('published demo.jpg\n')
            rate.sleep()

if __name__ == '__main__':
    try:
        worker = SegmentationWorker()
        rospy.init_node("instance_segmentation_node", anonymous=True)
        worker.publish()
    except rospy.ROSInterruptException:
        pass
