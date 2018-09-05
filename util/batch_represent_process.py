# -*- coding: utf8 -*-
# ! /usr/bin/python

import tensorlayer as tl
from multiprocessing import Process,Queue,Lock
from src import face
import pickle
import os
import cv2,sys
import numpy as np
import tensorflow as tf
from scipy import misc
sys.path.append('./src/')
import align.detect_face
import facenet
import argparse
import signal
from tqdm import tqdm
from tensorflow.python.ops import data_flow_ops
gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../models/20180518-142558"
import time
from six.moves import xrange
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
class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
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
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

class Encoder:
    def __init__(self):
        self.detect = Detection()
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding_array(self, images):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        nrof_samples = len(images)
        faceimages = np.zeros((nrof_samples, 160, 160, 3))
        for i in range(nrof_samples):
            faces = self.detect.find_faces(images[i])
            img = facenet.prewhiten(faces[0].image)
            faceimages[i, :, :, :] = img
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: faceimages, phase_train_placeholder: False}
        emb_array = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array



class FaceEncoder(Process):
#class FaceRecognition(threading.Thread):
    def __init__(self,q,faceq,facelock,output_dir):
        Process.__init__(self)
        #threading.Thread.__init__(self)
        self.thread_state = True
        self.face_recognition = None
        self.q = q
        self.faceq = faceq
        self.facelock = facelock
        self.output_dir = output_dir

    def run(self):
        self.q.put(os.getpid())
        self.encoder = Encoder()
        embedding_size = 128
        batch_size = 100
        image_size = 160
        num = 0
        while True:
            imagepaths = self.faceq.get()
            img_list = sorted(tl.files.load_file_list(path=self.data_dir, regx='.*.png', printable=False))
            imgs = tl.vis.read_images(img_list, path=self.data_dir, n_threads=32)
            nrof_images = len(img_list)
            nrof_batches = int(np.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((len(imagepaths), embedding_size))
            paths_array = []
            images_bar = tqdm(range(nrof_batches), desc='[{}:processing images saving result ]'.format(os.getpid()))
            for i in images_bar:
                # for i in xrange(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = img_list[start_index:end_index]
                images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False,
                                           image_size=image_size, do_prewhiten=True)
                emb_array = self.encoder.generate_embedding_array(images)


            np.save(os.path.join(self.output_dir, "{}_{}_signatures.npy".format((os.getpid(),num)), emb_array))
            labels_name_array = []
            for i in range(len(imagepaths)):
                labels_name_array += paths_array[i]
            np.save(os.path.join(self.output_dir, "{}_{}_labels_name.npy".format((os.getpid(),num)), labels_name_array))



def main_process(args):
    signal.signal(signal.SIGINT, sigint_handler)
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    procnum = 2
    load_mode_finish_q = Queue()
    classifier_mode_finish_q = Queue()
    face_msg_queue = Queue()

    reload_class_q = [Queue(), Queue()]

    facelock = Lock()
    face_proc = []
    for i in range(procnum):
        face_proc.append(
            FaceEncoder(load_mode_finish_q, face_msg_queue, facelock, output_dir))
    for i in range(procnum):
        face_proc[i].start()

    # while load_mode_finish_q.qsize()<procnum:
    # print(load_mode_finish_q.get())
    #    time.sleep(1)
    for i in range(len(face_proc)):
        load_mode_finish_q.get()




    while True:
        print(face.get_classifier_model_filenames('models'), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        print(classifier_mode_finish_q.get())
        for i in range(procnum):
            reload_class_q[i].put(i)


    for i in range(len(face_proc)):
        face_proc[i].join()

def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs, pubs, face_proc
    pubs.stop()
    subs.stop()
    for i in range(len(face_proc)):
        face_proc[i].stop()
    sys.exit()

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_images = 100000
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # load the model
            print("Loading trained model...\n")
            facenet.load_model(args.trained_model_dir)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            img_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='.*.png', printable=False))
            imgs = tl.vis.read_images(img_list, path=args.data_dir, n_threads=32)
            nrof_images = len(img_list)
            nrof_batches = int(np.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((save_images, embedding_size))
            paths_array = []
            images_bar = tqdm(range(nrof_batches), desc='[{}:processing images saving result ]'.format(os.getpid()))
            for i in images_bar:
            #for i in xrange(nrof_batches):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_array += img_list[start_index:end_index]
                align_imgs = tl.prepro.threading_data(imgs[start_index:end_index], fn=crop_sub_imgs_fn,
                                                      is_random=True)
                feed_dict = {images_placeholder: align_imgs, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                if i%save_images == 0:
                    np.save(os.path.join(output_dir, "signatures.npy"), emb_array)
                    labels_name_array = []
                    for i in range(len(save_images)):
                        labels_name_array += paths_array[i]
                    np.save(os.path.join(output_dir, "labels_name.npy"), labels_name_array)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Batch-represent face embeddings from a given data directory")
    parser.add_argument('-d', '--data_dir', type=str,
                        help='directory of images with structure as seen at the top of this file.')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='directory containing aligned face patches with file structure as seen at the top of this file.')
    parser.add_argument('--trained_model_dir', type=str,
                        help='Load a trained model before training starts.')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=50)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))