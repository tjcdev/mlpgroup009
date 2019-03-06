'''
 def set_model_params(self, params):
        with self.g.as_default():
            trainable_vars = tf.trainable_variables()
            idx = 0
            for var in trainable_vars:
                t_shape = self.sess.run(var).shape
                p_ = np.array(params[idx])
                assert t_shape == p_.shape, "inconsistent shape"
                assign_op = var.assign(p_.astype(np.float)/10000.)
                self.sess.run(assign_op)
                idx += 1

'''

import gym
import baselines from deepq

env = gym.make("LunarLander-v2")

