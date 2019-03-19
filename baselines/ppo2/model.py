import tensorflow as tf
import functools

import sys

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, load_path, skip_layers=[], frozen_weights=[], transfer_weights=False, microbatch_size=None):
        self.sess = sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        def print_weights(params):
            variables_names = [v.name for v in params]
            values = sess.run(variables_names)
            for k, v in zip(variables_names, values):
                if str(k) == 'ppo2_model/vf/w:0':
                    print("Variable: " + str(k))
                    print("Shape: " + str(v.shape))
                    print(v)

        # Transfer weights from an already trained model
        # TODO: this is if we are going to use transfer learning
        if transfer_weights:
            # Get all variables from the model.
            variables_to_restore = {v.name.split(":")[0]: v
                                    for v in tf.get_collection(
                                        tf.GraphKeys.GLOBAL_VARIABLES)}
                                    
            # Skip some variables during restore.
            skip_pretrained_var = skip_layers

            variables_to_restore = {
                v: variables_to_restore[v] for
                v in variables_to_restore if not
                any(x in v for x in skip_pretrained_var)}

            already_inits = variables_to_restore

            # Restore the remaining variables
            if variables_to_restore:
                saver_pre_trained = tf.train.Saver(
                    var_list=variables_to_restore)
                    
                saver_pre_trained.restore(sess, tf.train.latest_checkpoint(load_path))

            # Collect all trainale variables
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            # Freeze certain variables
            params = tf.contrib.framework.filter_variables(
                    params,
                    include_patterns=['model'],
                    exclude_patterns= frozen_weights)

            # Initialise all the other variables
            '''
            """Initialize all the uninitialized variables in the global scope."""
            new_variables = set(tf.global_variables())
            new_variables = tf.contrib.framework.filter_variables(
                    new_variables,
                    include_patterns=[],
                    exclude_patterns= variables_to_restore)
            tf.get_default_session().run(tf.variables_initializer(new_variables))   
            '''
        else:
            # If we are not using transfer learning
            # 1. Get the model parameters
            params = tf.trainable_variables('ppo2_model')
      
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = zip(grads, var)
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        #self.save = functools.partial(save_variables, sess=sess)
        #self.load = functools.partial(load_variables, sess=sess)

        initialize(already_inits)


        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def save(self, save_path):
        """
        Save the model
        """
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):
        """
        Load the model
        """
        saver = tf.train.Saver()
        print('Loading ' + load_path)
        saver.restore(self.sess, load_path)

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

