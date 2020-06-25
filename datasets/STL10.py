import os
import tensorflow as tf
from constants import STL10_TF_DATADIR

slim = tf.contrib.slim


class STL10:

    SPLITS_TO_SIZES = {'train_unlabeled': 100000, 'train': 5000, 'test': 8000, 'train_fold_0': 4000,
                       'train_fold_1': 4000,
                       'train_fold_2': 4000, 'train_fold_3': 4000, 'train_fold_4': 4000, 'train_fold_5': 4000,
                       'train_fold_6': 4000, 'train_fold_7': 4000, 'train_fold_8': 4000, 'train_fold_9': 4000,
                       'test_fold_0': 1000, 'test_fold_1': 1000, 'test_fold_2': 1000, 'test_fold_3': 1000,
                       'test_fold_4': 1000, 'test_fold_5': 1000, 'test_fold_6': 1000, 'test_fold_7': 1000,
                       'test_fold_8': 1000, 'test_fold_9': 1000
                       }

    def __init__(self, split_name='train_unlabeled'):
        self.split_name = split_name
        self.reader = tf.TFRecordReader
        self.label_offset = 0
        self.is_multilabel = False
        self.data_dir = STL10_TF_DATADIR
        self.file_pattern = 'stl10_%s.tfrecord'
        self.num_classes = 10
        self.name = 'STL10'
        self.num_samples = self.SPLITS_TO_SIZES[split_name]

    def feature_description(self):
        keys_to_features = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        return keys_to_features

    def get_items_to_handlers(self):
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', shape=[96, 96, 3], channels=3),
            'height': slim.tfexample_decoder.Tensor('image/height'),
            'width': slim.tfexample_decoder.Tensor('image/width'),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }
        return items_to_handlers

    def get_trainset_labelled(self):
        return self.get_split('train')

    def get_trainset_unlabelled(self):
        return self.get_split('train_unlabeled')

    def get_testset(self):
        return self.get_split('test')

    def get_train_fold_id(self, fold_idx):
        return 'train_fold_{}'.format(fold_idx)

    def get_test_fold_id(self, fold_idx):
        return 'test_fold_{}'.format(fold_idx)

    def get_num_train_labelled(self):
        return self.SPLITS_TO_SIZES['train']

    def get_num_train_unlabelled(self):
        return self.SPLITS_TO_SIZES['train_unlabeled']

    def get_num_test(self):
        return self.SPLITS_TO_SIZES['test']

    def get_split_size(self, split_name):
        return self.SPLITS_TO_SIZES[split_name]

    def format_labels(self, labels):
        return slim.one_hot_encoding(labels, self.num_classes)

    def get_split(self, split_name, data_dir=None):
        """Gets a dataset tuple with instructions for reading ImageNet.
        Args:
          split_name: A train/eval split name.
          data_dir: The base directory of the dataset sources.
        Returns:
          A `Dataset` namedtuple.
        Raises:
          ValueError: if `split_name` is not a valid train/eval split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        if not data_dir:
            data_dir = self.data_dir

        tf_record_pattern = os.path.join(data_dir, self.file_pattern % split_name)
        data_files = tf.io.gfile.glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset at %s' % data_dir)
        print('data_files: {}'.format(data_files))

        raw_dataset = tf.data.TFRecordDataset(data_files)

        # Build the decoder
        feature_description = self.feature_description()

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        parsed_dataset = raw_dataset.map(_parse_function)

        return parsed_dataset

    def get_dataset(self):
        return self.get_split(self.split_name)