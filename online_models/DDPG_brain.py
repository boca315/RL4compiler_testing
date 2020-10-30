# -*- coding: utf-8 -*-
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from environment import Environment, single_keys, group_keys, group_items_keys




#####################  hyper parameters  ####################

MAX_EPISODES = 20
MAX_EP_STEPS = 500
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32



###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, var,option):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.option = option
        self.var = var

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,


        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'+self.option):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'+self.option):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'+self.option+'/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'+self.option+'/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+self.option+'/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+self.option+'/target')

        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(s, 1, activation=tf.nn.relu, name='l1', trainable=trainable)

            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 1 # 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################


GLOBAL_ENV_SCOPE = 'Global_Env'
env = Environment(GLOBAL_ENV_SCOPE)

s_dim = env.state_size
a_dim = 1 # env.action_space.high (1,) int
a_bound = np.array([5.]) # <class 'numpy.ndarray'> [2.]



var = 2  # control exploration
t1 = time.time()

ddpgs = []

for param in single_keys:
    ddpgs.append(DDPG(a_dim, s_dim, a_bound, var = var,option=param))
for group in group_keys:
    group_dir = group_items_keys[group]
    for param in group_dir.keys():
        ddpgs.append(DDPG(a_dim, s_dim, a_bound, var = var,option=param))


for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    t2 = time.time()
    if t2 - t1 >= 86400.0:  # 24h
        break
    for j in range(MAX_EP_STEPS):
        t2 = time.time()
        if t2 - t1 >= 86400.0:  # 24h
            break
        actions = []
        for ddpg in ddpgs:
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, ddpg.var), -1, 1)    # add randomness to action selection for exploration
            actions.append(a[0])
        s_, r, done, info = env.step(actions, env.fv_history)
        for i in range(len(ddpgs)):
            ddpgs[i].store_transition(s, actions[i], r, s_)

            if ddpgs[i].pointer > MEMORY_CAPACITY:
                ddpgs[i].var *= .9995    # decay the action randomness
                ddpgs[i].learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)