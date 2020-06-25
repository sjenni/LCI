import time
import numpy as np
import sys
import os
from datetime import datetime
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_ops
from utils import get_variables_to_train, montage_tf, get_checkpoint_path, average_gradients

from .TrainerBase import BaseTrainer


class ClassifierTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        BaseTrainer.__init__(self, *args, **kwargs)

    def learning_rate(self):
        # Learning rate schedule for linear classifier experiments
        num_train_steps = self.num_train_steps
        boundaries = [np.int64(num_train_steps * 0.08), np.int64(num_train_steps * 0.4),
                      np.int64(num_train_steps * 0.7), np.int64(num_train_steps * 0.9)]
        values = [0.1, 0.01, 0.002, 0.0004, 0.00008]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def build_model(self, batch_queue, tower, opt, scope):
        imgs_train, labels_train = batch_queue.get_next()
        imgs_train.set_shape([self.model.batch_size, ]+self.model.im_shape)

        if self.model.im_shape[-1] == 3:
            tf.summary.image('imgs/train', montage_tf(imgs_train, 4, 8), max_outputs=1)

        # Create the model
        reuse = True if (tower > 0) else None
        preds = self.model.linear_classifiers(imgs_train, self.dataset.num_classes, training=True, reuse=reuse)

        # Compute losses
        loss = self.model.linear_class_loss(scope, preds, self.dataset.format_labels(labels_train))
        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients on all the batch splits.
        weights = get_variables_to_train(self.train_scopes)
        grads = opt.compute_gradients(loss, weights)

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss, grads, {}

    def train_model(self, chpt_path):
        print('Restoring from: {}'.format(chpt_path))
        g = tf.Graph()
        with g.as_default():
            with tf.device('/cpu:0'):
                # Init global step
                self.global_step = tf.train.create_global_step()

                batch_queue = self.get_data_queue()
                opt = self.optimizer()

                # Calculate the gradients for each model tower.
                tower_grads = []
                loss = None
                layers = None
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_{}'.format(i)) as scope:
                                loss, grads, layers = self.build_model(batch_queue, i, opt, scope)
                                tower_grads.append(grads)
                grad = average_gradients(tower_grads)

                # Make summaries
                self.make_summaries(grad, layers)

                # Apply the gradients to adjust the shared variables.
                apply_gradient_op = opt.apply_gradients(grad, global_step=self.global_step)

                train_op = control_flow_ops.with_dependencies([apply_gradient_op], loss)

                # Create a saver.
                saver = tf.train.Saver(tf.global_variables())
                init_fn = self.make_init_fn(chpt_path)

                # Build the summary operation from the last tower summaries.
                summary_op = tf.summary.merge(self.summaries)

                # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

                # Start running operations on the Graph.
                sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False), graph=g)
                sess.run(init)
                prev_ckpt = get_checkpoint_path(self.get_save_dir())
                if prev_ckpt:
                    print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
                    saver.restore(sess, prev_ckpt)
                elif init_fn:
                    init_fn(sess)

                summary_writer = tf.summary.FileWriter(self.get_save_dir(), sess.graph)
                init_step = sess.run(self.global_step)
                print('Start training at step: {}'.format(init_step))
                for step in range(init_step, self.num_train_steps):

                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 50 == 0:
                        num_examples_per_step = self.model.batch_size * self.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / self.num_gpus
                        print('{}: step {}/{}, loss = {} ({} examples/sec; {} sec/batch)'
                              .format(datetime.now(), step, self.num_train_steps, loss_value,
                                      examples_per_sec, sec_per_batch))
                        sys.stdout.flush()

                    if step % (self.num_train_steps // self.num_summary_steps) == 0:
                        print('Writing summaries...')
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % (self.num_train_steps // self.num_summary_steps * 4) == 0 or (
                            step + 1) == self.num_train_steps:
                        checkpoint_path = os.path.join(self.get_save_dir(), 'model.ckpt')
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        saver.save(sess, checkpoint_path, global_step=step)
