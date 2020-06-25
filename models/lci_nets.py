from math import log
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from .sn_ops import snconv2d, snlinear


def get_normalizer(is_training):
    bn_fn_args = {
        'is_training': is_training,
        'fused': True,
        'scale': True,
        'decay': 0.975,
        'epsilon': 0.001,
    }
    return slim.batch_norm, bn_fn_args


def patch_discriminator(inputs,
                        depth=64,
                        update_collection=None,
                        reuse=tf.AUTO_REUSE,
                        num_layers=None,
                        scope='discriminator'):
    inp_shape = inputs.get_shape().as_list()[1]

    end_points = {}
    with tf.variable_scope(scope, values=[inputs], reuse=reuse):
        net = inputs
        if num_layers is None:
            num_layers = int(log(inp_shape, 2)) - 2
        for i in range(1, num_layers):
            current_depth = min(depth * 2 ** (i-1), 512)
            net = snconv2d(net, current_depth, k_h=3, k_w=3, d_h=1, d_w=1, name='conv%i_0' % (i + 1),
                           update_collection=update_collection)
            net = tf.nn.leaky_relu(net, 0.1)
            print('Discriminator layer {}_0: {}'.format(i, net.get_shape().as_list()))
            net = snconv2d(net, current_depth, k_h=4, k_w=4, d_h=2, d_w=2, name='conv%i_1' % (i + 1),
                           update_collection=update_collection)
            net = tf.nn.leaky_relu(net, 0.1)
            print('Discriminator layer {}_1: {}'.format(i, net.get_shape().as_list()))
            end_points[scope] = net

        current_depth = min(depth * 2 ** (num_layers-1), 512)
        net = snconv2d(net, current_depth, d_h=1, d_w=1, name='conv_{}'.format(num_layers),
                       update_collection=update_collection)
        net = tf.nn.leaky_relu(net, 0.1)
        print('Discriminator layer {}: {}'.format(num_layers, net.get_shape().as_list()))
        net = tf.reduce_sum(net, axis=[1, 2])
        print('Discriminator pre_FC: {}'.format(net.get_shape().as_list()))

        net = slim.flatten(net)
        logits = snlinear(net, 1, update_collection=update_collection)
        end_points['logits'] = logits

        return logits, end_points


def patch_inpainter(net,
                    depth=32,
                    num_outputs=3,
                    is_training=True,
                    num_layers=4,
                    reuse=tf.AUTO_REUSE,
                    encoder_scope='encoder_ae',
                    decoder_scope='decoder_ae'):
    normalizer_fn, normalizer_fn_args = get_normalizer(is_training)
    act_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)

    with tf.variable_scope(encoder_scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            stride=2,
                            kernel_size=4,
                            activation_fn=act_fn):
            net = slim.conv2d(net, depth, kernel_size=3, stride=1, normalizer_fn=None, scope='conv0')
            print('Encoder layer 0: {}'.format(net.get_shape().as_list()))
            for i in range(1, num_layers):
                scope = 'conv%i' % (i + 1)
                current_depth = min(depth * 2 ** (i), 512)
                net = slim.conv2d(net, current_depth, normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_fn_args, scope=scope)
                print('Encoder layer {}: {}'.format(i, net.get_shape().as_list()))

    with tf.variable_scope(decoder_scope, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d_transpose],
                            normalizer_fn=normalizer_fn,
                            stride=2,
                            kernel_size=4,
                            activation_fn=act_fn):

            for i in range(1, num_layers):
                scope = 'deconv%i' % (i)
                current_depth = min(depth * 2 ** (num_layers - 1 - i), 512)
                net = slim.conv2d_transpose(net, current_depth, normalizer_fn=normalizer_fn,
                                            normalizer_params=normalizer_fn_args, scope=scope)
                print('Decoder layer {}: {}'.format(i, net.get_shape().as_list()))

            # Last layer has different normalizer and activation.
            scope = 'deconv%i' % (num_layers + 1)
            net = slim.conv2d_transpose(net, num_outputs, kernel_size=3, stride=1,
                                        normalizer_fn=None, activation_fn=None, scope=scope)

            print('Decoder output: {}'.format(net.get_shape().as_list()))

            return tf.nn.tanh(net), None