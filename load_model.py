import gym
import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables


def load(sess, load_path):
    """
    Load the model
    """
    saver = tf.train.Saver()
    print('Loading ' + load_path)
    saver.restore(sess, load_path)

with tf.Session() as sess:
    model =  load(sess,'./baseline_weights/ppo2_bip')
