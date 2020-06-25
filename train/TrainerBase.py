import sys
import os
import tensorflow.compat.v1 as tf
from utils import remove_missing
from constants import LOG_DIR
import tensorflow.contrib.slim as slim


class BaseTrainer():
    def __init__(self, model, data_generator, pre_processor, num_epochs, optimizer='momentum', momentum=0.9,
                 lr_policy='const', init_lr=0.01, end_lr=None, num_gpus=1, train_scopes='encoder',
                 exclude_scopes=['global_step'], lr_decay=0.95, d_lr_mult=4.):
        self.model = model
        self.dataset = data_generator
        self.pre_processor = pre_processor
        self.num_epochs = num_epochs
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.d_lr_mult = d_lr_mult
        self.end_lr = end_lr if end_lr is not None else 0.01 * init_lr
        self.num_gpus = num_gpus
        self.num_summary_steps = 80
        self.summaries = []
        self.moving_avgs_decay = 0.999
        self.global_step = None
        self.train_scopes = train_scopes
        self.exclude_scopes = exclude_scopes
        self.num_train_steps = (self.dataset.num_samples // self.model.batch_size) * self.num_epochs
        self.num_train_steps //= self.num_gpus
        print('Number of training steps: {}'.format(self.num_train_steps))

    def get_data_queue(self):
        # Loading of training data
        data = self.dataset.get_dataset()
        data = data.repeat()
        data = data.shuffle(buffer_size=min(self.dataset.num_samples, 100000))
        data = data.map(self.pre_processor.process_train, num_parallel_calls=8)
        data = data.batch(self.model.batch_size)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = tf.data.make_one_shot_iterator(data)
        return iterator

    def preprocess(self, img, label):
        img = self.pre_processor.process_train(img)
        return img, label

    def make_init_fn(self, chpt_path):
        # Handle model initialization from prior checkpoint
        if chpt_path is None:
            return None

        var2restore = slim.get_variables_to_restore(exclude=self.exclude_scopes)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        var2restore = remove_missing(var2restore, chpt_path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(chpt_path, var2restore)
        sys.stdout.flush()

        # Create an initial assignment function.
        def init_fn(sess):
            print('Restoring from: {}'.format(chpt_path))
            sess.run(init_assign_op, init_feed_dict)

        return init_fn

    def get_save_dir(self):
            fname = '{}_{}'.format(self.model.name, self.dataset.name)
            return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self, type=None):
        lr = self.learning_rate()
        if type is 'd':
            lr *= self.d_lr_mult    # Use larger learning rate for discriminator
        opts = {'momentum': tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.momentum, use_nesterov=True),
                'adam': tf.train.AdamOptimizer(learning_rate=lr, beta1=self.momentum)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {
            'const': self.init_lr
        }
        return policies[self.lr_policy]

    def make_summaries(self, grads, layers):
        self.summaries.append(tf.summary.scalar('learning_rate', self.learning_rate()))
        # Variable summaries
        for variable in slim.get_model_variables():
            self.summaries.append(tf.summary.histogram(variable.op.name, variable))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                self.summaries.append(tf.summary.histogram('gradients/' + var.op.name, grad))
        # Add histograms for activation.
        if layers:
            for layer_id, val in layers.iteritems():
                self.summaries.append(tf.summary.histogram('activations/' + layer_id, val))
