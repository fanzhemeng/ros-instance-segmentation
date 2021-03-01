#!/usr/bin/env python3
from __future__ import print_function

import sys
import time
from threading import Thread, Lock
from queue import Queue
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from YolactEdgeEngine import YolactEdgeEngine, ImageResult
from waytous_perception_msgs.msg import Object, ObjectArray, Rect

class SegmentationWorker:

    def __init__(self):
        pass

    def setup(self):
        self.mBatchSize = 3
        self.mSubTopic01 = "/Cam/Image_raw01"
        self.mSubTopic02 = "/Cam/Image_raw02"
        self.mSubTopic03 = "/Cam/Image_raw03"

        self.mPubTopic01 = "/camera_01/det_info"
        self.mPubTopic02 = "/camera_02/det_info"
        self.mPubTopic03 = "/camera_03/det_info"
        self.mPubTopic04 = "/vis01"
        self.mPubTopic05 = "/vis02"
        self.mPubTopic06 = "/vis03"

        self.bridge = CvBridge()
        self.engine = YolactEdgeEngine()

        self.mMsgsIn01 = Queue()
        self.mMsgsIn02 = Queue()
        self.mMsgsIn03 = Queue()
        self.mMtx01 = Lock()
        self.mMtx02 = Lock()
        self.mMtx03 = Lock()

        self.mSub01 = rospy.Subscriber(self.mSubTopic01, Image, self.callback01, queue_size=1)
        print('subscribed to [%s]'% self.mSubTopic01)
        self.mSub02 = rospy.Subscriber(self.mSubTopic02, Image, self.callback02, queue_size=1)
        print('subscribed to [%s]'% self.mSubTopic02)
        self.mSub03 = rospy.Subscriber(self.mSubTopic03, Image, self.callback03, queue_size=1)
        print('subscribed to [%s]'% self.mSubTopic03)

        self.mPub01 = rospy.Publisher(self.mPubTopic01, ObjectArray, queue_size=1)
        self.mPub02 = rospy.Publisher(self.mPubTopic02, ObjectArray, queue_size=1)
        self.mPub03 = rospy.Publisher(self.mPubTopic03, ObjectArray, queue_size=1)
        self.mPub04 = rospy.Publisher(self.mPubTopic04, Image, queue_size=1)
        self.mPub05 = rospy.Publisher(self.mPubTopic05, Image, queue_size=1)
        self.mPub06 = rospy.Publisher(self.mPubTopic06, Image, queue_size=1)
        print('publish on [%s], [%s], [%s] and /vis0x' % (self.mPubTopic01, self.mPubTopic02, self.mPubTopic03))

        self.mIsWorking = True
        self.mSegmentThread = Thread(target=self.segment, args=())
        self.mSegmentThread.daemon = True
        self.mSegmentThread.start()

    def callback01(self, data):
        with self.mMtx01:
            self.mMsgsIn01.put(data)

    def callback02(self, data):
        with self.mMtx02:
            self.mMsgsIn02.put(data)

    def callback03(self, data):
        with self.mMtx03:
            self.mMsgsIn03.put(data)

    def segment(self):
        while self.mIsWorking:
            if not self.mMsgsIn01.empty() and not self.mMsgsIn02.empty() and not self.mMsgsIn03.empty():
                with self.mMtx01:
                    msgIn01 = self.mMsgsIn01.get()
                with self.mMtx02:
                    msgIn02 = self.mMsgsIn02.get()
                with self.mMtx03:
                    msgIn03 = self.mMsgsIn03.get()
                self.mCvImageIn = []
                try:
                    self.mCvImageIn.append(self.bridge.imgmsg_to_cv2(msgIn01, 'bgr8'))
                    self.mCvImageIn.append(self.bridge.imgmsg_to_cv2(msgIn02, 'bgr8'))
                    self.mCvImageIn.append(self.bridge.imgmsg_to_cv2(msgIn03, 'bgr8'))
                except CvBridgeError as e:
                    print(e)

                start_time = time.time()
                cvImgOut, allres = self.engine.detect(self.mCvImageIn, return_imgs=True)
                end_time = time.time()
                print('%.3f s' % (end_time-start_time))

                try:
                    msgMask01 = self.bridge.cv2_to_imgmsg(allres[0].mask)
                    msgMask02 = self.bridge.cv2_to_imgmsg(allres[1].mask)#np.zeros((1024,1280,1),dtype='uint8'))
                    msgMask03 = self.bridge.cv2_to_imgmsg(allres[2].mask)
                    msgImgOut01 = self.bridge.cv2_to_imgmsg(cvImgOut[0], 'bgr8')
                    msgImgOut02 = self.bridge.cv2_to_imgmsg(cvImgOut[1], 'bgr8')
                    msgImgOut03 = self.bridge.cv2_to_imgmsg(cvImgOut[2], 'bgr8')
                except CvBridgeError as e:
                    print(e)

                self.mMsgOut01 = ObjectArray()
                self.mMsgOut02 = ObjectArray()
                self.mMsgOut03 = ObjectArray()

                self.mMsgOut01.header = msgIn01.header
                self.mMsgOut02.header = msgIn02.header
                self.mMsgOut03.header = msgIn03.header

                self.mMsgOut01.image = msgMask01
                self.mMsgOut02.image = msgMask02
                self.mMsgOut03.image = msgMask03

                for i in range(self.mBatchSize):
                    res = allres[i]
                    print("batch, num_dets ", (i, res.num_dets))
                    for j in range(res.num_dets):
                        obj = Object()
                        r = Rect()
                        x1, y1, x2, y2 = res.boxes[j, :]
                        r.x = x1
                        r.y = y1
                        r.w = x2-x1
                        r.h = y2-y1
                        r.orientation = 0
                        obj.obb2d = r
                        obj.sub_type = res.classes[j]
                        obj.exist_prob = res.scores[j]
                        if i==0:
                            self.mMsgOut01.foreground_objects.append(obj)
                        elif i==1:
                            self.mMsgOut02.foreground_objects.append(obj)
                        elif i==2:
                            self.mMsgOut03.foreground_objects.append(obj)

                self.mPub01.publish(self.mMsgOut01)
                self.mPub02.publish(self.mMsgOut02)
                self.mPub03.publish(self.mMsgOut03)
                self.mPub04.publish(msgImgOut01)
                self.mPub05.publish(msgImgOut02)
                self.mPub06.publish(msgImgOut03)

                end_time = time.time()
                print('total: %.3f s' % (end_time-start_time))

if __name__ == '__main__':
    rospy.init_node("instance_segmentation_node", anonymous=True)
    worker = SegmentationWorker()
    worker.setup()
    print('setup complete. Detecting...')
    try:
        rospy.spin()
    except:
        pass
    print('Shutting down.')
