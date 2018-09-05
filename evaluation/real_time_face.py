# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
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

import pickle
import os
from FSRGAN import FSRGAN
from SRGAN import SRGAN
import cv2,sys
import numpy as np
import tensorflow as tf
from scipy import misc
sys.path.append('../')
import src.align.detect_face
import src.facenet
from TinyFace.tiny_face import Tinyface
import time
gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../models/20180519-210715"


def get_classifier_model_filenames(model_dir):
    files = os.listdir(model_dir)
    pkl_files = [s for s in files if s.endswith('.pkl')]
    if len(pkl_files) == 0:
        raise ValueError('No pkl file found in the model directory (%s)' % model_dir)

    pkl_files = sorted(pkl_files)
    return pkl_files[-1]

#classifier_model = os.path.dirname(__file__) + "/../models/KNN.pkl"
#classifier_model = get_classifier_model_filenames(os.path.dirname(__file__) + "/../models")
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.align_time = 0.0
        self.net_time = 0.0
        self.class_time = 0.0
        self.confidence = 0.0


class Recognition:
    def __init__(self):
        self.encoder = Encoder()
        self.identifier = Identifier()
        self.detect = Detection()
        #self.detect = TinyFaceDetection()


    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        start_time = time.time()

        faces = self.detect.find_faces(image)
        end_time = time.time()
        for i, face in enumerate(faces):
            face.align_time = end_time-start_time
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            start_time = time.time()
            face.embedding = self.encoder.generate_embedding(face)
            face.net_time = time.time()-start_time
            start_time = time.time()
            face.name,face.confidence = self.identifier.identify(face)
            face.class_time = time.time()-start_time

        return faces


class Identifier:
    def __init__(self):
        classifier_dir = os.path.dirname(__file__) + "/../models"
        classifier_filenames = get_classifier_model_filenames(classifier_dir)
        classifier_model =os.path.join(classifier_dir, 'KNN.pkl')
        print('load Classifier model:', classifier_model)
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)
            print('11111111111111111:',self.class_names)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            #predictions = self.model.predict([face.embedding])
            #print(predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            #print(best_class_indices)
            return self.class_names[best_class_indices[0]], predictions[0][best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            src.facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = src.facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]



class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=16):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.srgan = FSRGAN.LHFSRGAN()
        #self.srgan = SRGAN.LHSRGAN()

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return src.align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = src.align.detect_face.detect_face(image, self.minsize,
                                                              self.pnet, self.rnet, self.onet,
                                                              self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            if cropped.shape[0]<80 or cropped.shape[1]<80:
                cropped, _ = self.srgan.LR_to_HR(cropped)          


            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces


class TinyFaceDetection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=10):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.crop_scale = (face_crop_size-face_crop_margin)//face_crop_margin
        path = os.path.join(os.path.dirname(__file__), '../TinyFace/mat2tf.pkl')
        self.srgan = FSRGAN.LHFSRGAN()
        #self.srgan = SRGAN.LHSRGAN()

        self.tinyface = Tinyface(path)

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return src.align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):

        refined_bboxes = self.tinyface.evaluate(image)
        faces = []
        bounding_boxes, _ = src.align.detect_face.detect_face(image, self.minsize,
                                                              self.pnet, self.rnet, self.onet,
                                                              self.threshold, self.factor)
        for r in refined_bboxes:
            _r = [int(x) for x in r[:4]]
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)
            img_size = np.asarray(image.shape)[0:2]
            crop = self.face_crop_margin // (2 * self.crop_scale)
            face.bounding_box[0] = np.maximum(_r[0] - crop, 0)
            face.bounding_box[1] = np.maximum(_r[1] - crop, 0)
            face.bounding_box[2] = np.minimum(_r[2] + crop, img_size[1])
            face.bounding_box[3] = np.minimum(_r[3] + crop, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            if cropped.shape[0] < 80 or cropped.shape[1] < 80:
                cropped, _ = self.srgan.LR_to_HR(cropped)
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)

        return faces