import tensorflow as tf

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, src_shape=(256, 256)):
        self.target_shape = target_shape
        self.src_shape = src_shape

    def scale(self, image):
        image = tf.cast(image, tf.float32)/255.
        image = image * 2. - 1.
        image = tf.clip_by_value(image, -1., 1.)
        return image

    def process_train(self, example):
        image = tf.image.decode_jpeg(example['image_raw'])
        image = tf.image.random_crop(image, self.target_shape)
        image = tf.image.random_flip_left_right(image)
        image = self.scale(image)
        return image, example['label']

    def process_test(self, example):
        image = tf.image.decode_jpeg(example['image_raw'])
        dp = int((self.src_shape[0]-self.target_shape[0])/2)
        image = image[dp:dp+self.target_shape[0], dp:dp+self.target_shape[1], :]
        image = self.scale(image)
        return image, example['label']

    def process_test_multicrop(self, example):
        image = tf.image.decode_jpeg(example['image_raw'])
        image = self.scale(image)
        return image, example['label']
