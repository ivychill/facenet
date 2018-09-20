import argparse
import sys
import os, shutil
import numpy as np
import glob

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, subdir, name, image_paths, num):
        self.subdir = subdir
        self.name = name
        self.image_paths = image_paths
        self.num = num

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class ImageClass2():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    print(path_exp)
    paths_tmp = [path for path in os.listdir(path_exp) \
                 if os.path.isdir(os.path.join(path_exp, path))]
    paths_tmp.sort()
    nrof_paths = len(paths_tmp)
    for i in range(nrof_paths):
        path_exp2 = os.path.join(path_exp, paths_tmp[i])
        classes = [path for path in os.listdir(path_exp2) \
                   if os.path.isdir(os.path.join(path_exp2, path))]
        classes.sort()
        nrof_classes = len(classes)
        print(paths_tmp[i] + "have classes:" + str(nrof_classes))
        for j in range(nrof_classes):
            class_name = classes[j]
            facedir = os.path.join(path_exp2, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(paths_tmp[i], class_name, image_paths, (i + 1) * (j + 1)))
            print(paths_tmp[i], class_name, i * j)
    return dataset

def get_dataset_txt(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass2(class_name, image_paths))

    return dataset


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    HR_DIR = output_dir + '/HR'
    LANDMARK_DIR = output_dir + '/landmark'
    LABELS_DIR = output_dir + '/labels'
    if not os.path.exists(HR_DIR):
        os.makedirs(HR_DIR)
    if not os.path.exists(LANDMARK_DIR):
        os.makedirs(LANDMARK_DIR)
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)

    dataset = get_dataset(args.input_dir)
    for cls in dataset:
        output_labels_dir = os.path.join(LABELS_DIR, str(cls.num).zfill(5))
        if not os.path.exists(output_labels_dir):
            os.makedirs(output_labels_dir)

        HR_filename = args.input_dir + '/' + cls.subdir + '/' + cls.name + '.png'
        out_HR_filename = os.path.join(HR_DIR, str(cls.num).zfill(5) + '.png')
        shutil.copyfile(HR_filename, out_HR_filename)
        landmark_filename = args.input_dir + '/' + cls.subdir + '/' + cls.name + '.txt'
        out_landmark_filename = os.path.join(LANDMARK_DIR, str(cls.num).zfill(5) + '.txt')
        shutil.copyfile(landmark_filename, out_landmark_filename)
        cls.image_paths.sort()
        # for image_path in cls.image_paths:
        for i in range(len(cls.image_paths)):
            output_labels_filename = os.path.join(output_labels_dir,
                                                  str(cls.num).zfill(5) + '_lbl' + str(i).zfill(2) + '.png')
            shutil.copyfile(cls.image_paths[i], output_labels_filename)

def gen_pairs(path,cycle,num):
    dataset = get_dataset_txt(path)
    pairspath = os.path.join(path,'pairs.txt')
    file = open(pairspath, 'w')
    file.write(str(cycle) + '    ' + str(num) + '\n')
    #txt = []
    #txt.append(str(cycle) + '    ' + str(num) + '\n')
    for i in range(cycle):
        oo = 0
        while oo < num:
            if len(dataset) > 1:
                num_cls = np.random.randint(0, len(dataset))
                cls = dataset[num_cls]
                if len(cls.image_paths)>0:
                    # split source and target, source: *_00xx.png, target: *_01xx.png
                    id_images = [s for s in cls.image_paths if '_00' in s]
                    # glob.glob(cls.image_paths+'*_01??.png')
                    im_no1 = np.random.randint(0, len(id_images))
                    im_no2 = np.random.randint(len(id_images), len(cls.image_paths))
                    sort_paths = np.sort(cls.image_paths)
                    print(sort_paths[im_no1].split('.')[-2])
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    #no1 = im_no1
                    #no2 = im_no2
                    if im_no2 != im_no1:
                        #txt.append(cls.name + '        ' + str(im_no1) + '        ' + str(im_no2))
                        file.write(cls.name + '    ' + str(no1) + '    ' + str(no2) + '\n')
                        oo = oo + 1
        nn = 0
        while nn < num:
            cls_no1 = np.random.randint(0, len(dataset))
            cls_no2 = np.random.randint(0, len(dataset))
            if cls_no1 != cls_no2:
                #txt.append(cls1.name + '    ' + str(im_no1) + '    ' + cls2.name + '    ' + str(im_no2))
                cls1 = dataset[cls_no1]
                cls2 = dataset[cls_no2]
                if len(cls1.image_paths) > 0 and len(cls2.image_paths) > 0:
                    im_no1 = np.random.randint(0, len(cls1.image_paths))
                    sort_paths = np.sort(cls1.image_paths)
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    #no1 = im_no1
                    sort_paths = np.sort(cls2.image_paths)
                    im_no2 = np.random.randint(0, len(cls2.image_paths))
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    #no2 = im_no2
                    file.write(cls1.name + '    ' + str(no1) + '    ' + cls2.name + '    ' + str(no2) + '\n')
                    nn = nn + 1
        # file.write(''.join(txt))
    #print(len(txt))
    file.flush()
    file.close()

def main_txt(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if(args.mode=='pairs'):
        gen_pairs(args.input_dir, 10, 300)
    else:
        SELF_TRAIN_DIR = output_dir + '/train_align'
        SELF_VAL_DIR = output_dir + '/valid_align'
        #SELF_TRAIN_DIR = '/opt/liukang/face_data/makeface_A_01_out_train'
        #SELF_VAL_DIR = '/opt/liukang/face_data/makeface_A_01_out_test'
        if not os.path.exists(SELF_TRAIN_DIR):
            os.makedirs(SELF_TRAIN_DIR)
        if not os.path.exists(SELF_VAL_DIR):
            os.makedirs(SELF_VAL_DIR)
        #'''

        dataset = get_dataset_txt(args.input_dir)
        #dataset = get_dataset(args.input_dir)

        num = 0
        for cls in dataset:
            #if(len(cls.name)<5):
            #   cls.name = str(cls.name).zfill(5)
            # for image_path in cls.image_paths:

            train = 1
            val = 1
            if(len(cls.image_paths)>5):
                for i in range(len(cls.image_paths)):
                    if i < len(cls.image_paths)*2//3:                        
                        output_train_dir = os.path.join(SELF_TRAIN_DIR, cls.name)
                        if not os.path.exists(output_train_dir):
                            os.makedirs(output_train_dir)
                        output_filename = os.path.join(output_train_dir, cls.name + '_' + str(train).zfill(4) + '.png')
                        shutil.copyfile(cls.image_paths[i], output_filename)
                        train = train +1
                    else:
                        output_val_dir = os.path.join(SELF_VAL_DIR, cls.name)
                        if not os.path.exists(output_val_dir):
                            os.makedirs(output_val_dir)
                        output_filename = os.path.join(output_val_dir, cls.name + '_' + str(val).zfill(4) + '.png')
                        shutil.copyfile(cls.image_paths[i], output_filename)
                        val = val + 1
                num = num + 1
        gen_pairs(SELF_TRAIN_DIR,10,300)
        gen_pairs(SELF_VAL_DIR,10,300)
    #'''
    #gen_pairs('massiveSample',10,300)

def main_sh(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path=args.input_dir
    cycle = 10
    num = 100
    dataset = get_dataset_txt(path)
    pairspath = os.path.join(path, 'pairs.txt')
    file = open(pairspath, 'w')
    file.write(str(cycle) + '    ' + str(num) + '\n')
    # txt = []
    # txt.append(str(cycle) + '    ' + str(num) + '\n')
    for i in range(cycle):
        oo = 0
        while oo < num:
            if len(dataset) > 1:
                num_cls = np.random.randint(0, len(dataset))
                cls = dataset[num_cls]
                if len(cls.image_paths) > 0:
                    im_no1 = np.random.randint(0, 2)
                    im_no2 = np.random.randint(3, 5)
                    sort_paths = np.sort(cls.image_paths)
                    # print(sort_paths[im_no1].split('.')[-2])
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    # no1 = im_no1
                    # no2 = im_no2
                    if im_no2 != im_no1:
                        # txt.append(cls.name + '        ' + str(im_no1) + '        ' + str(im_no2))
                        file.write(cls.name + '    ' + str(no1) + '    ' + str(no2) + '\n')
                        oo = oo + 1
        nn = 0
        while nn < num:
            cls_no1 = np.random.randint(0, len(dataset))
            cls_no2 = np.random.randint(0, len(dataset))
            if cls_no1 != cls_no2:
                # txt.append(cls1.name + '    ' + str(im_no1) + '    ' + cls2.name + '    ' + str(im_no2))
                cls1 = dataset[cls_no1]
                cls2 = dataset[cls_no2]
                if len(cls1.image_paths) > 0 and len(cls2.image_paths) > 0:
                    im_no1 = np.random.randint(0, 2)
                    sort_paths = np.sort(cls1.image_paths)
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    # no1 = im_no1
                    sort_paths = np.sort(cls2.image_paths)
                    im_no2 = np.random.randint(3, 5)
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    # no2 = im_no2
                    file.write(cls1.name + '    ' + str(no1) + '    ' + cls2.name + '    ' + str(no2) + '\n')
                    nn = nn + 1
                    # file.write(''.join(txt))
    # print(len(txt))
    file.flush()
    file.close()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', default='/opt/yanhong.jia/datasets/lfw_align_160', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('--output_dir', default='.', type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('--mode', default='pairs', type=str,
                        help='gen_pairs.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    main_txt(parse_arguments(sys.argv[1:]))
    #main_sh(parse_arguments(sys.argv[1:]))
