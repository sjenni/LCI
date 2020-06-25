import tensorflow.compat.v1 as tf
from utils import get_checkpoint_path
import numpy as np


class ClassifierTester:
    def __init__(self, model, data_generator, pre_processor):
        self.model = model
        self.pre_processor = pre_processor
        self.data_generator = data_generator
        self.num_eval_steps = self.data_generator.num_samples//self.model.batch_size

    def get_data_queue(self):
        data = self.data_generator.get_dataset()
        data = data.map(self.pre_processor.process_test, num_parallel_calls=1)
        data = data.batch(self.model.batch_size)
        data = data.prefetch(100)
        iterator = tf.data.make_one_shot_iterator(data)
        return iterator

    def get_data_queue_multicrop(self):
        data = self.data_generator.get_dataset()
        data = data.map(self.pre_processor.process_test_multicrop, num_parallel_calls=1)
        data = data.batch(self.model.batch_size)
        data = data.prefetch(100)
        iterator = tf.data.make_one_shot_iterator(data)
        return iterator

    def preprocess(self, img, label):
        img = self.pre_processor.process_test(img)
        return img, label

    def make_test_summaries(self, names_to_values):
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)
        return summary_ops

    def test_classifier(self, ckpt_dir):
        print('Restoring from: {}'.format(ckpt_dir))

        g = tf.Graph()
        with g.as_default():
            # Get test batches
            batch_queue = self.get_data_queue()
            imgs_test, labels_test = batch_queue.get_next()
            imgs_test.set_shape([self.model.batch_size, ]+self.model.im_shape)

            # Get predictions
            predictions = self.model.linear_classifiers(imgs_test, self.data_generator.num_classes, training=False)

            num_corrects_list = []
            for preds, f_id in zip(predictions, self.model.feats_IDs):
                preds_test = tf.argmax(preds, 1)
                correct_preds = tf.equal(preds_test, labels_test)
                num_correct = tf.reduce_sum(tf.to_float(correct_preds))
                num_corrects_list.append(num_correct)

            # Start running operations on the Graph.
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            prev_ckpt = get_checkpoint_path(ckpt_dir)
            print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, prev_ckpt)

            n_cor_np = np.zeros([len(self.model.feats_IDs)])
            for i in range(self.num_eval_steps):
                n_correct = sess.run(num_corrects_list)
                n_cor_np += n_correct

            acc = n_cor_np/self.data_generator.num_samples
            print('Accuracy: {}'.format(acc))

            return acc

    def test_classifier_multicrop(self, ckpt_dir):
        print('Restoring from: {}'.format(ckpt_dir))

        g = tf.Graph()
        with g.as_default():
            # Get test batches
            batch_queue = self.get_data_queue_multicrop()
            imgs_test, labels_test = batch_queue.get_next()
            imgs_test.set_shape((self.model.batch_size,) + self.pre_processor.src_shape + (3,))
            print('imgs_test: {}'.format(imgs_test.get_shape().as_list()))

            # Extract crops
            imgs_rcrop = []
            dp = int((self.pre_processor.src_shape[0] - self.pre_processor.target_shape[0]) / 2)
            imgs_ccrop = imgs_test[:, dp:dp + self.pre_processor.target_shape[0], dp:dp + self.pre_processor.target_shape[1], :]
            imgs_rcrop.append(imgs_ccrop)
            imgs_ulcrop = imgs_test[:, :self.pre_processor.target_shape[0], :self.pre_processor.target_shape[1], :]
            imgs_rcrop.append(imgs_ulcrop)
            imgs_urcrop = imgs_test[:, :self.pre_processor.target_shape[0], -self.pre_processor.target_shape[1]:, :]
            imgs_rcrop.append(imgs_urcrop)
            imgs_blcrop = imgs_test[:, -self.pre_processor.target_shape[0]:, :self.pre_processor.target_shape[1], :]
            imgs_rcrop.append(imgs_blcrop)
            imgs_brcrop = imgs_test[:, -self.pre_processor.target_shape[0]:, -self.pre_processor.target_shape[1]:, :]
            imgs_rcrop.append(imgs_brcrop)
            imgs_rcrop_stack = tf.concat(imgs_rcrop, 0)

            # Add flipped crops
            imgs_rcrop_stack = tf.concat([imgs_rcrop_stack, tf.reverse(imgs_rcrop_stack, [2])], 0)

            preds_rcrop_stack = self.model.linear_classifiers(imgs_rcrop_stack, self.data_generator.num_classes, training=False)

            num_corrects_list = []
            for preds_stack, f_id in zip(preds_rcrop_stack, self.model.feats_IDs):
                stack_preds = tf.stack(tf.split(preds_stack, 10))
                stack_preds = tf.nn.softmax(stack_preds, axis=-1)
                preds = tf.reduce_mean(stack_preds, 0)
                preds_test = tf.argmax(preds, 1)
                correct_preds = tf.equal(preds_test, labels_test)
                num_correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                num_corrects_list.append(num_correct)

            # Start running operations on the Graph.
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            prev_ckpt = get_checkpoint_path(ckpt_dir)
            print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, prev_ckpt)

            n_cor_np = np.zeros([len(self.model.feats_IDs)])
            for i in range(self.num_eval_steps):
                n_correct = sess.run(num_corrects_list)
                n_cor_np += n_correct

            acc = n_cor_np/self.data_generator.num_samples
            print('Accuracy: {}'.format(acc))

            return acc
