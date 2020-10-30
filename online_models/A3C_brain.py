# -*- coding: utf-8 -*-
import multiprocessing
import threading
# import tensorflow as tf
import numpy as np
import os,psutil
import shutil
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from environment import Environment

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 1 #multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
GLOBAL_ENV_SCOPE = 'Global_Env'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

check_mem = True

# env = gym.make(GAME)
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.shape[0]
# A_BOUND = [env.action_space.low, env.action_space.high]

low = np.array([0.00], dtype="float32")
high = np.array([100.00], dtype="float32")
GLOBAL_ENV = Environment(GLOBAL_ENV_SCOPE)  # global env saves the config vectors
N_S = len(GLOBAL_ENV.reset())
N_A = 1 #10
A_BOUND = [low, high]


class ACNet(object):
    def __init__(self, scope, param, globalAC=None):
        self.param = param
        # print(scope)
        if GLOBAL_NET_SCOPE in scope:   # get global network
            # print("global")
            with tf.variable_scope(scope):
                self.low = np.array([0.00], dtype="float32")
                self.high = np.array([100.00], dtype="float32")
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
                # print(type(self.a_params))
        else:   # local net, calculate losses
            # print("non-global")
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)
                # print("mu:{}, sigma:{}, v: {}, a-param: {}, c-param: {:.2} "
                      # .format(mu, sigma, self.v, self.a_params, self.c_params))

                # # Tensor("W_1/actor/mu/Tanh:0", shape=(?, 1), dtype=float32, device=/device:CPU:0)
                # # Tensor("W_1/actor/sigma/Softplus:0", shape=(?, 1), dtype=float32, device=/device:CPU:0)
                # # Tensor("W_1/critic/v/BiasAdd:0", shape=(?, 1), dtype=float32, device=/device:CPU:0)
                # print(mu)
                # print(sigma)
                # print(self.v)

                # # <tf.Variable 'W_1/actor/sigma/bias:0' shape=(x, x) dtype=float32_ref>
                # # [shape=(71, 200), shape=(200,), shape=(200, 1), shape=(1,), shape=(200, 1), shape=(1,)]
                # print(self.a_params)
                # # [shape=(71, 100), shape=(100,), shape=(100, 1), shape=(1,) ]
                # print(self.c_params)


                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add c_loss:', process.memory_info().rss / 1024 / 1024, 'MB')

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add wrap_a_out:', process.memory_info().rss / 1024 / 1024, 'MB')

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add a_loss:', process.memory_info().rss / 1024 / 1024, 'MB')

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add choose_a:', process.memory_info().rss / 1024 / 1024, 'MB')

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add local_grad:', process.memory_info().rss / 1024 / 1024, 'MB')

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add sync pull:', process.memory_info().rss / 1024 / 1024, 'MB')

                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('add sync push:', process.memory_info().rss / 1024 / 1024, 'MB')



    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('actor net:', process.memory_info().rss / 1024 / 1024, 'MB')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('critic net:', process.memory_info().rss / 1024 / 1024, 'MB')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        # print("self s is",self.s,type(self.s))
        # print("s is", s,s.shape)
        # print("A is",self.A,type(self.A))
        return SESS.run(self.A, {self.s: s})



class Worker(object):
    # def __init__(self, name, globalACs, env, global_env):
    def __init__(self, name, globalACs, env):

        # self.env = gym.make(GAME).unwrapped
        self.env = env
        self.name = name
        self.ACs = globalACs #
        # self.global_env = global_env
        # print(self.global_env.fv_history)
        # for param in list(self.env.base_itm_single.keys()):
        #     self.ACs.append(ACNet(name, param, globalAC))
        # for group in self.env.base_itm_group.keys():
        #     group_dir = self.env.base_itm_group[group]
        #     for param in group_dir.keys():
        #         self.ACs.append(ACNet(name, param, globalAC))


    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            # print("reset ", len(s))
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()

                # print("abound low",env.action_space.low,type(env.action_space.low))
                # print("abound",A_BOUND,type(A_BOUND))
                # print("self low",self_bod[0],type(self_bod[0]))
                # print("self ",self_bod,type(self_bod))

                # print("gym state",s,s.shape)
                # print("self s",Environment().reset(),Environment().reset().shape)
                # print("ready s ",len(s))
                # print("iter s ", len(s))
                actions = []
                for i in range(self.env.state_size):# 71
                    agent = self.ACs[i]
                    a = agent.choose_action(s)[0]
                    actions.append(a)
                    # print(i)
                print("actions is ")
                # print(actions)
                # s_, r, done, info = self.env.step(actions, self.global_env.fv_history)
                s_, r, done, info = self.env.step(actions, self.env.fv_history)
                print("complete a step")
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_r.append((r+8)/8)    # normalize
                print("add to buffer")


                for i in range(self.env.state_size):# 71
                    print("update agent "+str(i))
                    agent = self.ACs[i]
                    buffer_a.append(actions[i])
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                        if done:
                            v_s_ = 0  # terminal
                        else:
                            v_s_ = SESS.run(agent.v, {agent.s: s_[np.newaxis, :]})[0, 0]
                        buffer_v_target = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                            buffer_v_target)
                        feed_dict = {
                            agent.s: buffer_s,
                            agent.a_his: buffer_a,
                            agent.v_target: buffer_v_target,
                        }
                        print("update_global agent " + str(i))
                        agent.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        print("pull_global agent " + str(i))
                        agent.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        # "info",info,
                        # "done",done,
                        # "s_",s_,
                        # "rwd",r,
                        "action",actions,
                        "state space",N_S,self.env.vals
                        # "action space",N_A,type(env.action_space),
                        # "boundary",type(A_BOUND)
                          )
                    res_train = open('train.txt', 'a')
                    res_train.flush()
                    res_train.write(self.name+
                        " Ep:"+GLOBAL_EP +
                        "| Ep_r: " + str(GLOBAL_RUNNING_R[-1])+
                        # "info",info,
                        # "done",done,
                        # "s_",s_,
                        "reward"+str(r)
                        # "action" + actions,
                        # "state space",N_S,self.env.vals
                        # "action space",N_A,type(env.action_space),
                        # "boundary",type(A_BOUND)
                          )
                    res_train.close()

                    GLOBAL_EP += 1
                    break



if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_ACs = [] # we only need its params
        # GLOBAL_ENV = Environment(GLOBAL_NET_SCOPE)
        # GLOBAL_ACs.append(ACNet(GLOBAL_NET_SCOPE, 'more_struct_union_type_prob') )

        for param in list(GLOBAL_ENV.base_itm_single.keys()):
            GLOBAL_ACs.append(ACNet(GLOBAL_NET_SCOPE+param, param))
        for group in GLOBAL_ENV.base_itm_group.keys():
            group_dir = GLOBAL_ENV.base_itm_group[group]
            for param in group_dir.keys():
                GLOBAL_ACs.append(ACNet(GLOBAL_NET_SCOPE+param, param))

        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            # env_i = Environment(i_name)
            ACs = []
            index = 0
            # ACs.append(ACNet(i_name, 'more_struct_union_type_prob', GLOBAL_ACs[0]))
            for param in list(GLOBAL_ENV.base_itm_single.keys()):
                if (check_mem):
                    process = psutil.Process(os.getpid())
                    print('agent'+str(index) +' mem:', process.memory_info().rss / 1024 / 1024, 'MB')
                ACs.append(ACNet(i_name+param, param, GLOBAL_ACs[index]))
                index += 1
            for group in GLOBAL_ENV.base_itm_group.keys():
                group_dir = GLOBAL_ENV.base_itm_group[group]
                for param in group_dir.keys():
                    if (check_mem):
                        process = psutil.Process(os.getpid())
                        print('agent' + str(index) + ' mem:', process.memory_info().rss / 1024 / 1024, 'MB')
                    ACs.append(ACNet(i_name+param, param, GLOBAL_ACs[index]))
                    index += 1
            if (check_mem):
                process = psutil.Process(os.getpid())
                print('worker' + str(i) + ' mem:', process.memory_info().rss / 1024 / 1024, 'MB')
            workers.append(Worker(i_name, ACs,  GLOBAL_ENV))

    print("done append worker")

    COORD = tf.train.Coordinator()
    if (check_mem):
        process = psutil.Process(os.getpid())
        print('tf.train.Coordinator mem:', process.memory_info().rss / 1024 / 1024, 'MB')

    SESS.run(tf.global_variables_initializer())
    if (check_mem):
        process = psutil.Process(os.getpid())
        print('SESS.run mem:', process.memory_info().rss / 1024 / 1024, 'MB')


    print("done SESS RUN")


    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)


    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('add a job mem:', process.memory_info().rss / 1024 / 1024, 'MB')
        t = threading.Thread(target=job)
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('new a thread mem:', process.memory_info().rss / 1024 / 1024, 'MB')
        t.start()
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('start the thread mem:', process.memory_info().rss / 1024 / 1024, 'MB')
        worker_threads.append(t)
        if (check_mem):
            process = psutil.Process(os.getpid())
            print('append the thread mem:', process.memory_info().rss / 1024 / 1024, 'MB')

    print("done add threads")
    COORD.join(worker_threads)
    print("done join threads")


    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()

