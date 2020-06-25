import argparse
import sys
import tensorflow as tf
from Preprocessor import Preprocessor
from train.TrainerSSL import CINTrainer
from train.TrainerLinearClassifier import ClassifierTrainer
from eval.ClassifierTester import ClassifierTester
from datasets.STL10 import STL10
from models.TCNet import TRCNet
from utils import write_experiments_multi, wait_for_new_checkpoint

# Basic model parameters as external flags.
FLAGS = None


def main(_):
    # Define data pre-processors
    load_shape = [80, 80, 3]
    shape_transfer = [64, 64, 3]
    crop_sz = (64, 64)
    preprocessor = Preprocessor(target_shape=load_shape, src_shape=(96, 96, 3))
    preprocessor_lin = Preprocessor(target_shape=shape_transfer, src_shape=(96, 96, 3))

    # Initialize the data generators
    data_gen_ssl = STL10('train_unlabeled')
    data_gen_ftune = STL10('train')
    data_test = STL10('test')

    # Define the network and SSL training
    model = TRCNet(batch_size=FLAGS.batch_size, im_shape=load_shape, n_tr_classes=6, tag=FLAGS.tag,
                   lci_patch_sz=42, lci_crop_sz=48, n_layers_lci=4, ae_dim=48,
                   enc_params={'padding': 'SAME'})
    trainer = CINTrainer(model=model, data_generator=data_gen_ssl, pre_processor=preprocessor, crop_sz=crop_sz,
                         wd_class=FLAGS.wd, init_lr_class=FLAGS.pre_lr,
                         num_epochs=FLAGS.n_eps_pre, num_gpus=FLAGS.num_gpus,
                         optimizer='adam', init_lr=0.0002, momentum=0.5,  # Parameters for LCI training only
                         train_scopes='features')
    trainer.train_model(None)

    # Get the final checkpoint
    ckpt_dir_model = trainer.get_save_dir()
    ckpt = wait_for_new_checkpoint(ckpt_dir_model, last_checkpoint=None)
    print('Found checkpoint: {}'.format(ckpt))
    ckpt_id = ckpt.split('-')[-1]

    # Train linear classifiers on frozen features
    tag_class = '{}_classifier_ckpt_{}'.format(FLAGS.tag, ckpt_id)
    model = TRCNet(batch_size=FLAGS.batch_size_ftune, im_shape=shape_transfer, tag=tag_class,
                   feats_ids=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                   enc_params={'use_fc': False, 'padding': 'SAME'})
    trainer_class = ClassifierTrainer(model=model, data_generator=data_gen_ftune, pre_processor=preprocessor_lin,
                                      optimizer='momentum', init_lr=FLAGS.ftune_lr, momentum=0.9,
                                      num_epochs=FLAGS.n_eps_ftune, num_gpus=1,
                                      train_scopes='classifier')
    trainer_class.train_model(ckpt)
    ckpt_dir = trainer_class.get_save_dir()

    # Evaluate on the test set
    model.batch_size = 100
    tester = ClassifierTester(model=model, data_generator=data_test, pre_processor=preprocessor_lin)
    acc = tester.test_classifier(ckpt_dir)
    write_experiments_multi(acc, tag_class, FLAGS.tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_ftune', type=int, default=64)
    parser.add_argument('--n_eps_pre', type=int, default=200,
                        help='Number of epochs for pre-training.')
    parser.add_argument('--n_eps_ftune', type=int, default=900,
                        help='Number of epochs for transfer learning.')
    parser.add_argument('--pre_lr', type=float, default=3e-4,
                        help='Initial learning rate for pre-training.')
    parser.add_argument('--ftune_lr', type=float, default=0.1,
                        help='Initial learning rate for transfer learning.')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)