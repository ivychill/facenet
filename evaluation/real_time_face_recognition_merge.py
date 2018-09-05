# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time

import cv2

import real_time_face as face


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        roi_x = 0
        roi_y = 0
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)

            rows, cols = face.image.shape[:2]
            img2 = face.image
            roi = frame[roi_y:roi_y+cols,roi_x:roi_x+rows]  # 创建掩膜
            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)  # 保留除图像外的背景
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            dst = cv2.add(img1_bg, img2)  # 进行融合
            frame[roi_y:roi_y + cols, roi_x:roi_x + rows] = dst  # 融合后放在原图上

            if face.name is not None:
                print('confidence:{}~name:{}'.format(face.confidence, face.name))
                if (face.confidence < 0.6):
                    face.name = "unknown" + '({})'.format(face.confidence)
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
                cv2.putText(frame, face.name, (roi_x + rows, roi_y + cols//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                            thickness=2, lineType=2)
            roi_y = roi_y + cols
            if (roi_y > frame.shape[0]):
                roi_x = roi_x + cols
                roi_y = 0




    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    frame_interval = 5  # Number of frames after which to run face detection
    fps_display_interval = 2  # seconds
    frame_rate = 0
    frame_count = 0

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('/opt/yanhong.jia/datasets/facevideo/20180520095819.ts')
    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        cv2.imshow('Video', cv2.resize(frame,(1920,1080)))
        #cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
