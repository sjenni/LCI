import tensorflow.compat.v1 as tf
from tensorflow.python.ops import control_flow_ops
from .TrainerBase import BaseTrainer
import numpy as np
import sys
import os
import time
from datetime import datetime
from utils import get_variables_to_train, montage_tf, get_checkpoint_path, weights_montage, average_gradients
import tensorflow.contrib as contrib
import tensorflow.contrib.slim as slim


class CINTrainer(BaseTrainer):

    def __init__(self, crop_sz, init_lr_class=3e-4, wd_class=1e-4, beta_1_class=0.5,
                 warp_factor=16., n_warp_points=16, *args, **kwargs):
        BaseTrainer.__init__(self, *args, **kwargs)
        self.crop_sz = crop_sz
        self.wd = wd_class
        self.init_lr_class = init_lr_class
        self.beta_1 = beta_1_class
        self.warp_factor = warp_factor
        self.n_warp_points = n_warp_points
        self.excl_gamma_wd = True
        self.excl_beta_wd = True
        self.weight_summary = True

    def build_model(self, batch_queue, opt_g, opt_d, opt_c, scope, tower_id):
        # Define scopes for LCI components
        lci_enc_scope = 'encoder_ae_{}'.format(tower_id)
        lci_dec_scope = 'decoder_ae_{}'.format(tower_id)
        lci_disc_scope = 'discriminator_{}'.format(tower_id)

        # Load batch of images
        imgs_train, _ = batch_queue.get_next()
        imgs_train.set_shape([self.model.batch_size, ] + self.model.im_shape)

        # Create warped images
        w_mag = self.model.im_shape[0]/self.warp_factor
        p_x = tf.random_uniform([self.model.batch_size, self.n_warp_points], minval=0, maxval=self.model.im_shape[0])
        p_y = tf.random_uniform([self.model.batch_size, self.n_warp_points], minval=0, maxval=self.model.im_shape[1])
        c_points_src = tf.stack([p_x, p_y], axis=-1)
        c_points_dest = c_points_src + tf.random_uniform(c_points_src.get_shape(), -w_mag, w_mag)
        imgs_warp, _ = contrib.image.sparse_image_warp(imgs_train, c_points_src, c_points_dest)
        tf.summary.image('imgs/img_warp', montage_tf(imgs_warp, 4, 8), max_outputs=1)

        # Perform LCI
        patch_lci, patch_ae, mask_erase, mask_orig, crop_img, imgs_lci, imgs_patchae =\
            self.model.lci(imgs_train, enc_scope=lci_enc_scope, dec_scope=lci_dec_scope)
        tf.summary.image('imgs/real_imgs', montage_tf(imgs_patchae, 4, 8), max_outputs=1)
        tf.summary.image('imgs/fake_imgs', montage_tf(imgs_lci, 4, 8), max_outputs=1)

        # Build untransformed images (half original, half with autoencoded patches)
        imgs_nt_1, _ = tf.split(imgs_train, 2)
        _, imgs_nt_2 = tf.split(imgs_patchae, 2)
        imgs_nt = tf.concat([imgs_nt_1, imgs_nt_2], 0)

        # Perform additional augmentations to make detection of LCI harder
        imgs_lci = random_crop_rot(imgs_lci, self.crop_sz)
        imgs_nt = random_crop_rot(imgs_nt, self.crop_sz)
        imgs_warp = random_crop_rot(imgs_warp, self.crop_sz)

        # Generate the rotated images
        imgs_rot, _ = all_rot(imgs_nt)

        # Patch disciminator for LCI
        preds_fake = self.model.patch_disc(patch_lci, update_collection="NO_OPS", disc_scope=lci_disc_scope)
        preds_real = self.model.patch_disc(crop_img, update_collection=None, disc_scope=lci_disc_scope)

        # The transformation classifier
        class_in = tf.concat([imgs_nt, imgs_lci, imgs_rot, imgs_warp], 0)
        preds = self.model.net(class_in)

        # Build SSL labels
        labels = tf.concat([tf.zeros((self.model.batch_size,), dtype=tf.int32),
                            tf.ones((self.model.batch_size,), dtype=tf.int32),
                            2 * tf.ones((self.model.batch_size,), dtype=tf.int32),
                            3 * tf.ones((self.model.batch_size,), dtype=tf.int32),
                            4 * tf.ones((self.model.batch_size,), dtype=tf.int32),
                            5 * tf.ones((self.model.batch_size,), dtype=tf.int32)], 0)
        labels = tf.one_hot(labels, 6)

        # Compute losses
        loss_c = self.model.loss_ssl(preds, labels)
        loss_disc_lci = self.model.discriminator_loss(preds_fake, preds_real)
        loss_ae_lci = self.model.inpainter_loss(preds_fake, crop_img, patch_lci, mask_erase, patch_ae, mask_orig)
        loss_ae_lci -= self.model.loss_lci_adv(preds, labels)

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss_c = control_flow_ops.with_dependencies([updates], loss_c)

        # Calculate the gradients for the batch of data on this tower.
        grads_d = opt_d.compute_gradients(loss_disc_lci, get_variables_to_train(lci_disc_scope))
        grads_g = opt_g.compute_gradients(loss_ae_lci, get_variables_to_train('{},{}'.format(lci_enc_scope, lci_dec_scope)))
        grads_c = opt_c.compute_gradients(loss_c, get_variables_to_train(self.train_scopes, print_vars=True))

        # Create some summaries
        if self.weight_summary:
            with tf.variable_scope('features', reuse=True):
                weights_disc_1 = slim.variable('conv_1/weights')
            tf.summary.image('weights/conv_1', weights_montage(weights_disc_1, 6, 16),
                             max_outputs=1)
        tf.summary.scalar('lr_decay_mult', self.lr_decay_mult())
        self.summaries += tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        return loss_ae_lci, loss_disc_lci, loss_c, grads_g, grads_d, grads_c, {}

    def optimizer_class(self):
        # Defines the optimizer for the transformation classifier
        decay = self.lr_decay_mult()
        lr = self.init_lr_class*decay
        wd = self.wd*decay
        return contrib.opt.AdamWOptimizer(wd, learning_rate=lr, beta1=self.beta_1, beta2=0.99)

    def lr_decay_mult(self):
        # Defines the cosine decay for the learning rate
        num_train_steps = (self.dataset.num_samples / self.model.batch_size) * self.num_epochs
        num_train_steps /= self.num_gpus
        lr = tf.train.cosine_decay(1., self.global_step, num_train_steps, alpha=1e-3)
        return lr

    def train_model(self, chpt_path=None):
        if chpt_path:
            print('Restoring from: {}'.format(chpt_path))
        g = tf.Graph()
        with g.as_default():
            with tf.device('/cpu:0'):
                # Init global step
                self.global_step = tf.train.create_global_step()

                # Init data
                batch_queue = self.get_data_queue()

                # Optimizer for the classifier
                opt_c = self.optimizer_class()

                # Calculate the gradients for each model tower.
                train_ops_g = []
                train_ops_d = []
                tower_grads_c = []
                loss_c = 0.
                loss_g = 0.
                loss_d = 0.

                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            # LCI parameters are not shared across GPUs
                            opt_g = self.optimizer('g')
                            opt_d = self.optimizer('d')

                            with tf.name_scope('tower_{}'.format(i)) as scope:
                                l_g, l_d, l_c, grad_g, grad_d, grad_c, layers_d = \
                                    self.build_model(batch_queue, opt_g, opt_d, opt_c, scope, i)
                                loss_c += l_c
                                loss_g += l_g
                                loss_d += l_d

                            # Training ops for LCI
                            train_op_g = opt_g.apply_gradients(grad_g)
                            train_op_d = opt_d.apply_gradients(grad_d)
                            train_ops_d.append(train_op_d)
                            train_ops_g.append(train_op_g)

                            # Aggregate gradients for the transformation classifier
                            tower_grads_c.append(grad_c)

                # Average gradients for classifier from all GPUs
                grad_c = average_gradients(tower_grads_c)

                # Make summaries
                self.make_summaries(grad_d + grad_g + grad_c, layers_d)

                # Apply the gradients to adjust the shared variables.
                wd_vars = get_variables_to_train(self.train_scopes)
                if self.excl_gamma_wd:
                    wd_vars = [v for v in wd_vars if 'gamma' not in v.op.name]
                if self.excl_beta_wd:
                    wd_vars = [v for v in wd_vars if 'beta' not in v.op.name]
                print('WD variables: {}'.format([v.op.name for v in wd_vars]))
                train_op_c = opt_c.apply_gradients(grad_c, global_step=self.global_step, decay_var_list=wd_vars)

                # Group all updates to into a single train op.
                train_op = control_flow_ops.with_dependencies(
                    [train_op_c] + train_ops_d + train_ops_g, loss_d + loss_g + loss_c)

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
                    _, loss_value = sess.run([train_op, loss_c])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % (self.num_train_steps // 2000) == 0:
                        num_examples_per_step = self.model.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration
                        print('{}: step {}/{}, loss = {} ({} examples/sec; {} sec/batch)'
                              .format(datetime.now(), step, self.num_train_steps, loss_value,
                                      examples_per_sec, sec_per_batch))
                        sys.stdout.flush()

                    if step % (self.num_train_steps // 200) == 0:
                        print('Writing summaries...')
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % (self.num_train_steps // 40) == 0 or (step + 1) == self.num_train_steps:
                        checkpoint_path = os.path.join(self.get_save_dir(), 'model.ckpt')
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        saver.save(sess, checkpoint_path, global_step=step)


def all_rot(inp):
    rot_1 = tf.image.rot90(inp, 1)
    rot_2 = tf.image.rot90(inp, 2)
    rot_3 = tf.image.rot90(inp, 3)
    rots = tf.concat([rot_1, rot_2, rot_3], 0)
    return rots, None


def rand_rot(img):
    im_shape = tf.shape(img)
    bsz = im_shape[0]
    angles = tf.random_uniform([bsz, ], -np.pi * 0.07, np.pi * 0.07)
    img = contrib.image.rotate(img, angles, interpolation='BILINEAR')
    return img


def random_crop_rot(img, crop_sz=(64, 64), rot_split=2):
    im_shape = tf.shape(img)
    bsz = im_shape[0]

    img_1, img_2 = tf.split(img, [bsz-bsz//rot_split, bsz//rot_split])
    img_1 = rand_rot(img_1)
    img = tf.concat([img_1, img_2], 0)

    dx = (im_shape[1]-crop_sz[0])//2
    dy = (im_shape[2]-crop_sz[1])//2

    base = tf.constant(
        [1, 0, 0, 0, 1, 0, 0, 0], shape=[1, 8], dtype=tf.float32
    )
    base = tf.tile(base, [bsz, 1])

    mask_x = tf.constant(
        [0, 0, 1, 0, 0, 0, 0, 0], shape=[1, 8], dtype=tf.float32
    )
    mask_x = tf.tile(mask_x, [bsz, 1])

    mask_y = tf.constant(
        [0, 0, 0, 0, 0, 1, 0, 0], shape=[1, 8], dtype=tf.float32
    )
    mask_y = tf.tile(mask_y, [bsz, 1])

    jit_x = tf.random_uniform([bsz, 8], minval=-dx+1, maxval=dx, dtype=tf.int32)
    jit_x = tf.cast(jit_x, tf.float32)

    jit_y = tf.random_uniform([bsz, 8], minval=-dy+1, maxval=dy, dtype=tf.int32)
    jit_y = tf.cast(jit_y, tf.float32)

    xforms = base + jit_x * mask_x + jit_y*mask_y
    processed_data = contrib.image.transform(
        images=img, transforms=xforms
    )
    cropped_data = processed_data[:, dx:-dx, dy:-dy, :]
    return cropped_data