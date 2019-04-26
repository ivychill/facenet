
import tensorflow as tf
import numpy as np
from scipy import misc
import skimage
import cv2

def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
          image = tf.identity(image)
      return image


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  for i in range(num_cases):
      x=func(x,i)
  return x


def random_noise_image(image):
    image = skimage.util.random_noise(image, mode="gaussian")
    image *= 255
    image = np.uint8(image)
    return image
    #return skimage.util.random_noise(image, mode="gaussian")
    img = Image.fromarray(image)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(img)


def random_blur_image(image):
    return cv2.blur(image, (5, 5))
    img = Image.fromarray(image)
    img = img.filter(ImageFilter.BLUR)
    return np.array(img)
    return misc.imfilter(image,'blur')


def get_augmentation_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def augment_data(image, args, data_augmentation):
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # # image = tf.image.random_hue(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # image = tf.py_func(random_noise_image, [image], tf.uint8)
    # image = tf.py_func(random_blur_image, [image], tf.uint8)
    RANDOM_NOISE = 1 << 0
    RANDOM_BLUR = 1 << 1
    RANDOM_CROP = 1 << 2
    RANDOM_FLIP = 1 << 3
    RANDOM_COLOR = 1 << 4

    # image = tf.cond(get_augmentation_flag(data_augmentation[0], RANDOM_CROP),
    #                 lambda: tf.random_crop(image, [args.image_size, args.image_size, 3]),
    #                 lambda: tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size))
    image = tf.cond(get_augmentation_flag(data_augmentation[0], RANDOM_FLIP),
                    lambda: tf.image.random_flip_left_right(image),
                    lambda: tf.identity(image))
    image = tf.cond(get_augmentation_flag(data_augmentation[0], RANDOM_NOISE),
                    lambda: tf.py_func(random_noise_image, [image], tf.uint8),
                    lambda: tf.identity(image))
    image = tf.cond(get_augmentation_flag(data_augmentation[0], RANDOM_BLUR),
                    lambda: tf.py_func(random_blur_image, [image], tf.uint8),
                    lambda: tf.identity(image))
    image = tf.cond(get_augmentation_flag(data_augmentation[0], RANDOM_COLOR),
                    lambda: apply_with_random_selector(image, lambda x, ordering: distort_color(x, ordering),
                                                       num_cases=4),
                    lambda: tf.identity(image))
    return image