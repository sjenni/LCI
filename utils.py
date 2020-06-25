import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
import numpy as np
import time
import os


def write_experiments_multi(acc, id, tag):
    fname = 'results_multi_acc_{}.txt'.format(tag)
    with open(fname, 'a') as f:
        line = 'Tag: {} Acc:{}\n'.format(id, acc)
        f.write(line)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def montage_tf(imgs, num_h, num_w):
    """Makes a montage of imgs that can be used in image_summaries.

    Args:
        imgs: Tensor of images
        num_h: Number of images per column
        num_w: Number of images per row

    Returns:
        A montage of num_h*num_w images
    """
    imgs = tf.unstack(imgs)
    img_rows = [None] * num_h
    for r in range(num_h):
        img_rows[r] = tf.concat(axis=1, values=imgs[r * num_w:(r + 1) * num_w])
    montage = tf.concat(axis=0, values=img_rows)
    return tf.expand_dims(montage, 0)


def remove_missing(var_list, model_path):
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
        var_dict = var_list
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:

        if reader.has_tensor(var):
            available_vars[var] = var_dict[var]
        else:
            logging.warning(
                'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
    return var_list


def get_variables_to_train(trainable_scopes=None, print_vars=False):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

    if print_vars:
        print('Variables to train: {}'.format([v.op.name for v in variables_to_train]))

    return variables_to_train


def get_checkpoint_path(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if not ckpt:
        print("No checkpoint in {}".format(checkpoint_dir))
        return None
    return ckpt.model_checkpoint_path


def weights_montage(weights, grid_Y, grid_X, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
        weights: tensor of shape [Y, X, NumChannels, NumKernels]
        (grid_Y, grid_X): shape of the grid. Require: NumKernels == grid_Y * grid_X
        pad: number of black pixels around each filter (between them)

    Return:
        Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    x_min = tf.reduce_min(weights, axis=[0, 1, 2])
    x_max = tf.reduce_max(weights, axis=[0, 1, 2])

    weights1 = (weights - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(weights1 - 1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT') + 1

    # X and Y dimensions, w.r.t. padding
    Y = weights1.get_shape()[0] + 2 * pad
    X = weights1.get_shape()[1] + 2 * pad

    channels = weights1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
    """Waits until a new checkpoint file is found.
    Args:
      checkpoint_dir: The directory in which checkpoints are saved.
      last_checkpoint: The last checkpoint path used or `None` if we're expecting
        a checkpoint for the first time.
      seconds_to_sleep: The number of seconds to sleep for before looking for a
        new checkpoint.
      timeout: The maximum amount of time to wait. If left as `None`, then the
        process will wait indefinitely.
    Returns:
      a new checkpoint path, or None if the timeout was reached.
    """
    logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_path = ckpt.model_checkpoint_path
        ckpt_id = checkpoint_path.split('/')[-1]
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_id)

        # checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or checkpoint_path == last_checkpoint:
            if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
                return None
            time.sleep(seconds_to_sleep)
        else:
            logging.info('Found new checkpoint at %s', checkpoint_path)

            return checkpoint_path
