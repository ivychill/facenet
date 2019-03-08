
import os
import facenet
import tensorflow as tf


def get_supervised_dataset(path):
    domain_supervised_dataset = {}
    path_exp = os.path.expanduser(path)
    domains = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    domains.sort()

    def insert_image_paths(class_name,image_paths):
        for key,value in domain_supervised_dataset.items():
            for cls in value:
                if class_name == cls.name:
                    cls.image_paths += image_paths
                    return True
        return False

    for domain_name in domains:
        if domain_name != "id+camera":
        #if True:
            dataset = []
            path_dir_exp = os.path.join(path_exp, domain_name)
            classes = [path for path in os.listdir(path_dir_exp) \
                       if os.path.isdir(os.path.join(path_dir_exp, path))]
            classes.sort()
            nrof_classes = len(classes)
            for i in range(nrof_classes):
                class_name = classes[i]
                facedir = os.path.join(path_dir_exp, class_name)
                image_paths = facenet.get_image_paths(facedir)

                # print(domains)
                if insert_image_paths(class_name,image_paths) is False:
                    dataset.append(facenet.ImageClass(class_name, image_paths))
            if len(dataset)>0:
                domain_supervised_dataset[domain_name] = dataset

    return domain_supervised_dataset


def get_unsupervised_dataset(path):
    domain_unsupervised_dataset = {}
    path_exp = os.path.expanduser(path)
    domains = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    domains.sort()

    for domain_name in domains:
        if domain_name != "id+camera":
            facedir = os.path.join(path_exp, domain_name)
            image_paths = facenet.get_image_paths(facedir)
            for i in range(len(image_paths) - 1, -1, -1):  # for i in range(0, num_list.__len__())[::-1]
                extname = os.path.splitext(os.path.split(image_paths[i])[1])[1]
                if extname not in ['.jpg', '.png']:
                    image_paths.pop(i)

            path_dir_exp = os.path.join(path_exp, domain_name)
            classes = [path for path in os.listdir(path_dir_exp) \
                       if os.path.isdir(os.path.join(path_dir_exp, path))]
            classes.sort()
            nrof_classes = len(classes)
            for i in range(nrof_classes):
                class_name = classes[i]
                facedir = os.path.join(path_dir_exp, class_name)
                image_paths += facenet.get_image_paths(facedir)

            domain_unsupervised_dataset[domain_name] = facenet.ImageClass(domain_name, image_paths)

    return domain_unsupervised_dataset


def get_dataset(path):
    supervised_dataset = get_supervised_dataset(path)
    unsupervised_datset = get_unsupervised_dataset(path)
    return supervised_dataset, unsupervised_datset


def create_input_pipeline(input_queue, args, batch_size_placeholder):
    images_and_labels = []
    for _ in range(args.nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)

            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)


            # pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            images.append(tf.image.per_image_standardization(image))
        images_and_labels.append([images, label])

    image_batch, labels_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size_placeholder,
        shapes=[(args.image_size, args.image_size, 3), ()],
        enqueue_many=True,
        capacity=4 * args.nrof_preprocess_threads * args.batch_size,
        allow_smaller_final_batch=True)
    return image_batch, labels_batch


def create_domain_input_pipeline(input_queue, args, batch_size_placeholder):
    images_list = []
    for _ in range(args.nrof_preprocess_threads):
        source_filenames, target_filenames = input_queue.dequeue()
        source_images = []
        target_images = []
        for filename in tf.unstack(source_filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)
            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)

            # pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size) + (3,))
            source_images.append(tf.image.per_image_standardization(image))

        for filename in tf.unstack(target_filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)
            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)

            # pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size) + (3,))
            target_images.append(tf.image.per_image_standardization(image))
        images_list.append([source_images, target_images])

    source_image_batch, target_image_batch = tf.train.batch_join(
        images_list, batch_size=tf.div(batch_size_placeholder,2),
        shapes=[(args.image_size, args.image_size) + (3,), (args.image_size, args.image_size) + (3,)],
        enqueue_many=True,
        capacity=4 * args.nrof_preprocess_threads * args.batch_size,
        allow_smaller_final_batch=True)
    return source_image_batch, target_image_batch