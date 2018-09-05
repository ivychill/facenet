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
import tensorlayer as tl


import face
import  os
import datetime
import pandas as pd
def main(args):

    face_recognition = face.Recognition()

    correctPrediction = 0
    inCorrectPrediction = 0
    sumConfidence = 0.0

    if args.debug:
        print("Debug enabled")
        face.debug = True

    path_exp = os.path.expanduser(args.input_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    results = {'person_name': [], 'p_person_name': [], 'ailgn_time': [], 'network_time': [], 'predictions_time': [],
               'confidence': []}
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        valid_lr_img_list = sorted(tl.files.load_file_list(path=facedir, regx='.*.png', printable=False))
        valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=facedir, n_threads=4)
        for im in valid_lr_imgs:
            faces = face_recognition.identify(im)
            results['ailgn_time'].append(faces[0].align_time)
            results['network_time'].append(faces[0].net_time)
            results['predictions_time'].append(faces[0].class_time)
            results['confidence'].append(faces[0].confidence)
            results['person_name'].append(class_name)
            results['p_person_name'].append(faces[0].name)
            if class_name == faces[0].name :
                correctPrediction += 1
            else:
                inCorrectPrediction += 1
    Accuracy = float(correctPrediction) / (correctPrediction + inCorrectPrediction)
    Avg_Confidence = float(sumConfidence) / (correctPrediction + inCorrectPrediction)
    results['ailgn_time'].append('correctPrediction:' + str(correctPrediction))
    results['network_time'].append('inCorrectPrediction:' + str(inCorrectPrediction))
    results['predictions_time'].append('Accuracy:' + str(Accuracy))
    results['confidence'].append('Avg_Confidence:' + str(Avg_Confidence))
    results['person_name'].append('accuracy')
    results['p_person_name'].append('accuracy')
    dataname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    data_frame = pd.DataFrame(
        data={'person_name': results['person_name'], 'p_person_name': results['p_person_name'],
              'ailgn_time': results['ailgn_time'],
              'network_time': results['network_time'], 'predictions_time': results['predictions_time'],
              'confidence': results['confidence']})
    data_frame.to_csv(args.input_dir + '/results_' + dataname + '.csv')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/yiqi.liu-4/yanhong.jia/datasets/SELFDATA/test_align_160', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
