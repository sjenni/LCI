from .alexnet import alexnet_V2
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from utils import montage_tf
from .lci_nets import patch_inpainter, patch_discriminator
import tensorflow.contrib as contrib


# Average pooling params for imagenet linear classifier experiments
AVG_POOL_PARAMS = {'conv_1': (6, 6, 'SAME'), 'conv_2': (4, 4, 'VALID'), 'conv_3': (3, 3, 'SAME'),
                   'conv_4': (3, 3, 'SAME'), 'conv_5': (2, 2, 'VALID')}


class TRCNet:
    def __init__(self, batch_size, im_shape, n_tr_classes=6, lci_patch_sz=64, lci_crop_sz=80, ae_dim=48, n_layers_lci=5,
                 tag='default', feats_ids=None, feat_pool='AVG', enc_params=None):
        if enc_params is None:
            enc_params = {}
        self.name = 'TRNet_{}'.format(tag)
        self.n_tr_classes = n_tr_classes
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.feats_IDs = feats_ids
        self.feat_pool = feat_pool
        self.enc_params = enc_params
        self.lci_patch_sz = lci_patch_sz
        self.lci_crop_sz = lci_crop_sz
        self.num_LCI_layers = n_layers_lci
        self.ae_model = patch_inpainter
        self.class_model = alexnet_V2
        self.disc_model = patch_discriminator
        self.ae_dim = ae_dim

    def lci(self, img, enc_scope, dec_scope):
        # Extract random patch
        patch, jit_x, jit_y = random_crop(img, crop_sz=(self.lci_crop_sz, self.lci_crop_sz))

        # Erase the center of the patch
        patch_erased, mask_erase = patch_erase(patch, patch_sz=(self.lci_patch_sz, self.lci_patch_sz))
        tf.summary.image('imgs/patch_erased', montage_tf(patch_erased, 4, 8), max_outputs=1)

        # Perform inpainting/autoencoding
        net_in = tf.concat([patch, patch_erased], 0)
        net_out, _ = self.ae_model(net_in, depth=self.ae_dim, num_layers=self.num_LCI_layers,
                                   encoder_scope=enc_scope, decoder_scope=dec_scope)
        patch_ae, patch_ip = tf.split(net_out, 2)

        # Paste inpainted patches
        pasted_patch_inpaint, patch_mask = paste_crop(img, patch_ip, jit_x, jit_y)
        pasted_patch_ae, _ = paste_crop(img, patch_ae, jit_x, jit_y)
        img_lci = img * (1. - patch_mask) + pasted_patch_inpaint
        img_patchae = img * (1. - patch_mask) + pasted_patch_ae

        return patch_ip, patch_ae, mask_erase, tf.ones_like(mask_erase), patch, img_lci, img_patchae

    def ssl_net(self, net, reuse=None, training=True, scope='encoder'):
        return self.class_model(net, self.n_tr_classes, reuse, training, scope, **self.enc_params)

    def net(self, img, reuse=tf.AUTO_REUSE, training=True):
        preds, _ = self.ssl_net(img, reuse, training, scope='features')
        return preds

    def linear_classifiers(self, img, num_classes, training, reuse=None):
        _, feats = self.ssl_net(img, training=False, scope='features')

        preds_list = []
        with tf.variable_scope('classifier', reuse=reuse):
            for feats_id in self.feats_IDs:
                p = AVG_POOL_PARAMS[feats_id]
                if self.feat_pool == 'AVG':
                    class_in = slim.avg_pool2d(feats[feats_id], p[0], p[1], p[2])
                elif self.feat_pool == 'None':
                    class_in = feats[feats_id]

                print('{} linear classifier input shape: {}'.format(feats_id, class_in.get_shape().as_list()))
                preds = linear_classifier(class_in, num_classes, reuse, training, scope=feats_id, wd=5e-4)
                preds_list.append(preds)

        return preds_list

    def patch_disc(self, input, update_collection, disc_scope):
        in_1, in_2 = tf.split(input, 2)
        input = tf.concat([in_1, in_2], -1)
        model, _ = self.disc_model(input,
                                   update_collection=update_collection,
                                   num_layers=self.num_LCI_layers - 1,
                                   scope=disc_scope)
        return model

    def linear_class_loss(self, scope, preds, labels):
        total_loss = 0.

        for pred, f_id in zip(preds, self.feats_IDs):
            loss = tf.losses.softmax_cross_entropy(labels, pred, scope=scope)
            tf.summary.scalar('losses/SCE_{}'.format(f_id), loss)
            total_loss += loss

            # Compute accuracy
            predictions = tf.argmax(pred, 1)
            tf.summary.scalar('accuracy/train_accuracy_{}'.format(f_id),
                              slim.metrics.accuracy(predictions, tf.argmax(labels, 1)))

        loss_wd = tf.add_n(tf.losses.get_regularization_losses(), name='loss_wd')
        tf.summary.scalar('losses/loss_wd', loss_wd)

        total_loss = total_loss + loss_wd

        return total_loss

    def inpainter_loss(self, preads_fake, imgs, recs_erase, mask_erase, recs_orig, mask_orig):
        loss_fake = -tf.reduce_mean(preads_fake)
        tf.summary.scalar('losses/generator_fake_loss', loss_fake)

        loss_ae_erase = tf.losses.mean_squared_error(imgs, recs_erase, weights=50. * mask_erase)
        loss_ae_orig = tf.losses.mean_squared_error(imgs, recs_orig, weights=50. * mask_orig)

        tf.summary.scalar('losses/loss_ae_erase', loss_ae_erase)
        tf.summary.scalar('losses/loss_ae_orig', loss_ae_orig)

        return loss_fake + loss_ae_erase + loss_ae_orig

    def discriminator_loss(self, preds_fake, preds_real):
        loss_real = tf.reduce_mean(tf.nn.relu(1. - preds_real))
        loss_fake = tf.reduce_mean(tf.nn.relu(1. + preds_fake))

        loss = loss_real + loss_fake

        tf.summary.scalar('losses/disc_fake_loss', loss_fake)
        tf.summary.scalar('losses/disc_real_loss', loss_real)
        tf.summary.scalar('losses/disc_total_loss', loss)
        return loss

    def loss_ssl(self, preds, labels):
        # Define the loss
        loss = tf.losses.softmax_cross_entropy(labels, preds)
        tf.summary.scalar('losses/SCE', loss)

        # Compute accuracy
        predictions = tf.argmax(preds, 1)
        tf.summary.scalar('accuracy/train_accuracy',
                          slim.metrics.accuracy(predictions, tf.argmax(labels, 1)))

        bs = self.batch_size
        tf.summary.scalar('accuracy/train_accuracy_real_noae',
                          slim.metrics.accuracy(predictions[:bs // 2], tf.argmax(labels[:bs // 2], 1)))
        tf.summary.scalar('accuracy/train_accuracy_real_ae',
                          slim.metrics.accuracy(predictions[bs // 2:bs], tf.argmax(labels[bs // 2:bs], 1)))
        tf.summary.scalar('accuracy/train_accuracy_lci',
                          slim.metrics.accuracy(predictions[bs:2 * bs], tf.argmax(labels[bs:2 * bs], 1)))
        tf.summary.scalar('accuracy/train_accuracy_rot',
                          slim.metrics.accuracy(predictions[2 * bs:-bs], tf.argmax(labels[2 * bs:-bs], 1)))
        tf.summary.scalar('accuracy/train_accuracy_warp',
                          slim.metrics.accuracy(predictions[-bs:], tf.argmax(labels[-bs:], 1)))
        return loss

    def loss_lci_adv(self, preds, labels_tf):
        loss = tf.losses.softmax_cross_entropy(labels_tf, preds)
        return loss


def linear_classifier(net, num_out, reuse=None, training=True, scope='classifier', wd=5e-4):
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.batch_norm(net, decay=0.975, is_training=training, fused=True, center=False, scale=False)
        net = slim.flatten(net)
        net = slim.fully_connected(net, num_out,
                                   weights_initializer=contrib.layers.variance_scaling_initializer(),
                                   weights_regularizer=slim.l2_regularizer(wd),
                                   activation_fn=None, normalizer_fn=None)
        return net


def patch_erase(img, patch_sz=(16, 16)):
    im_shape = img.get_shape()
    pad_sz = [im_shape[1] - patch_sz[0], im_shape[2] - patch_sz[1]]
    patch_mask = tf.ones([im_shape[0], patch_sz[0], patch_sz[1], im_shape[3]])
    patch_mask = tf.pad(patch_mask,
                        [[0, 0], [pad_sz[0] // 2, pad_sz[0] // 2], [pad_sz[1] // 2, pad_sz[1] // 2], [0, 0]])
    return img * (1. - patch_mask) + 0.1 * patch_mask * tf.random_normal(im_shape), 1. - patch_mask


def random_crop(img, crop_sz=(20, 20)):
    im_shape = img.get_shape().as_list()
    bsz = im_shape[0]

    dx = (im_shape[1] - crop_sz[0]) // 2
    dy = (im_shape[2] - crop_sz[1]) // 2

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

    jit_x = tf.random_uniform([bsz, 8], minval=-dx + 1, maxval=dx, dtype=tf.int32)
    jit_x = tf.cast(jit_x, tf.float32)

    jit_y = tf.random_uniform([bsz, 8], minval=-dy + 1, maxval=dy, dtype=tf.int32)
    jit_y = tf.cast(jit_y, tf.float32)

    xforms = base + jit_x * mask_x + jit_y * mask_y
    processed_data = contrib.image.transform(
        images=img, transforms=xforms
    )
    cropped_data = processed_data[:, dx:dx + crop_sz[0], dy:dy + crop_sz[1], :]
    return cropped_data, jit_x, jit_y


def paste_crop(img, crop, jit_x, jit_y):
    im_shape = tf.shape(img)
    crop_shape = tf.shape(crop)

    bsz = im_shape[0]

    dx_1 = (im_shape[1] - crop_shape[1]) // 2
    dy_1 = (im_shape[2] - crop_shape[2]) // 2
    dx_2 = im_shape[1] - crop_shape[1] - dx_1
    dy_2 = im_shape[2] - crop_shape[2] - dy_1

    patch_mask = tf.ones_like(crop)
    crop = tf.pad(crop, [[0, 0], [dx_1, dx_2], [dy_1, dy_2], [0, 0]])
    patch_mask = tf.pad(patch_mask, [[0, 0], [dx_1, dx_2], [dy_1, dy_2], [0, 0]])

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

    xforms = base - jit_x * mask_x - jit_y * mask_y
    transformed_crop = contrib.image.transform(
        images=crop, transforms=xforms
    )
    transformed_mask = contrib.image.transform(
        images=patch_mask, transforms=xforms
    )

    return transformed_crop, transformed_mask
