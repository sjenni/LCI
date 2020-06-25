import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def scale_imgs(imgs):
    imgs = (imgs * 127.5) - tf.constant([[[[_R_MEAN - 127.5, _G_MEAN - 127.5, _B_MEAN - 127.5]]]],
                                        dtype=tf.float32, shape=(1, 1, 1, 3))
    return imgs


def conv_group(net, num_out, kernel_size, scope):
    input_groups = tf.split(axis=3, num_or_size_splits=2, value=net)
    output_groups = [slim.conv2d(j, num_out / 2, kernel_size=kernel_size, scope='{}/gconv{}'.format(scope, idx))
                     for (idx, j) in enumerate(input_groups)]
    return tf.concat(axis=3, values=output_groups)


# AlexNet
def alexnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True,
                     w_reg=0.00005, fix_bn=False):
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.975,
        'epsilon': 0.001,
        'center': True,
        'scale': True,
        'fused': train_bn,
    }
    normalizer_fn = slim.batch_norm

    with slim.arg_scope([slim.conv2d],
                        kernel_size=kernel_size,
                        padding=padding,
                        activation_fn=activation,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                return arg_sc


def alexnet_V2(net, num_out, reuse=tf.AUTO_REUSE, training=True, scope='encoder', padding='SAME', use_pool5=False,
               use_fc=True, scale_input=True):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(alexnet_argscope(training=training)):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

                if scale_input:
                    net = scale_imgs(net)
                feats = {}

                net = slim.conv2d(net, 96, kernel_size=[11, 11], stride=4, scope='conv_1', padding=padding)
                print('conv_1 feats: {}'.format(net.get_shape().as_list()))
                feats['conv_1'] = net

                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1', padding=padding)
                net = conv_group(net, 256, kernel_size=[5, 5], scope='conv_2')
                print('conv_2 feats: {}'.format(net.get_shape().as_list()))
                feats['conv_2'] = net

                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2', padding=padding)
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3')
                print('conv_3 feats: {}'.format(net.get_shape().as_list()))
                feats['conv_3'] = net

                net = conv_group(net, 384, kernel_size=[3, 3], scope='conv_4')
                print('conv_4 feats: {}'.format(net.get_shape().as_list()))
                feats['conv_4'] = net

                net = conv_group(net, 256, kernel_size=[3, 3], scope='conv_5')
                print('conv_5 feats: {}'.format(net.get_shape().as_list()))
                feats['conv_5'] = net

                if use_pool5:
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5', padding=padding)
                print('pre FC feats: {}'.format(net.get_shape().as_list()))

            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(0.0, 0.005)):

                enc_shape = net.get_shape().as_list()
                if use_fc:
                    net = slim.conv2d(net, 4096, kernel_size=enc_shape[1:3], padding='VALID', scope='fc_1')
                    feats['fc_1'] = net
                    net = slim.conv2d(net, 4096, kernel_size=[1, 1], padding='VALID', scope='fc_2')
                    feats['fc_2'] = net
                    net = slim.conv2d(net, num_out, kernel_size=[1, 1], padding='VALID', scope='fc_3',
                                      activation_fn=None, normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer())
                    net = slim.flatten(net)
        return net, feats
