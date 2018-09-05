import argparse
import sys
import os, shutil
import numpy as np
DATA_DIR = '/opt/release_v3.1/bench_marks_60_align'
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths

path_exp = os.path.expanduser(DATA_DIR)
classes = [path for path in os.listdir(path_exp) \
           if os.path.isdir(os.path.join(path_exp, path))]
classes.sort()
nrof_classes = len(classes)
for i in range(nrof_classes):
    class_name = classes[i]
    facedir = os.path.join(path_exp, class_name)
    image_paths = get_image_paths(facedir)
    image_paths.sort()
    for k in range(len(image_paths)):
        output_filename = os.path.join(facedir, facedir.split('/')[-1] + '_' + str(k + 1).zfill(4) + '.png')
        os.rename(image_paths[k], output_filename)

