#encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import time,random, os
import numpy as np
# from environment4 import Environment, single_keys, group_keys, group_items_keys, defaultConfig, N_PROGRAMS, seed, ACTION_BOUND
from .environment4 import Environment, single_keys, group_keys, group_items_keys,seed, ACTION_BOUND


random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class A2CNet(nn.Module):
    def __init__(self, s_dim, a_dim, name):
        super(A2CNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.name = name

        # ## structure 1 ##
        # self.pl1 = nn.Sequential(
        #     nn.Conv1d(s_dim,20, kernel_size=1, stride=2),
        #     # nn.BatchNorm1d(20),
        #     nn.ReLU()
        # )
        # self.pl2 = nn.Sequential(
        #     nn.Conv1d(20, 100, kernel_size=1),
        #     # nn.BatchNorm1d(100),
        #     nn.ReLU())
        # self.pfc = nn.Linear(100, a_dim)
        #
        # self.vl1 = nn.Sequential(
        #     nn.Conv1d(s_dim, 20, kernel_size=1, stride=2),
        #     # nn.BatchNorm1d(20),
        #     nn.ReLU())
        # self.vl2 = nn.Sequential(
        #     nn.Conv1d(20, 100, kernel_size=1),
        #     # nn.BatchNorm1d(100),
        #     nn.ReLU())
        # self.vfc = nn.Linear(100, 1)


        # ## structure 2 ##
        # self.pl = nn.Sequential(
        #     nn.Conv1d(s_dim, 20, kernel_size=1, stride=2),
        #     # nn.BatchNorm1d(20),
        #     nn.ReLU()
        # )
        # self.pfc = nn.Linear(20, a_dim)
        # self.vl = nn.Sequential(
        #     nn.Conv1d(s_dim, 20, kernel_size=1, stride=2),
        #     # nn.BatchNorm1d(20),
        #     nn.ReLU())
        # self.vfc = nn.Linear(20, 1)



        ## structure 3 ##
        self.pi1 = nn.Linear(s_dim, 20)
        self.pfc = nn.Linear(20, a_dim)
        self.v1 = nn.Linear(s_dim, 20)
        self.vfc = nn.Linear(20, 1)


        self.distribution = torch.distributions.Categorical



    def forward(self, s):
        # ## structure 1 ##
        # out = self.pl1(s) # 32, 50, 1
        # out = self.pl2(out) # 32, 100, 1
        # out = out.view(out.size(0), -1) # 32, 100 展平层
        # policy = self.pfc(out)
        # out = self.vl1(s)
        # out = self.vl2(out)
        # out = out.view(out.size(0), -1)
        # value = self.vfc(out)

        # ## structure 2 ##
        # out = self.pl(s)
        # out = out.view(out.size(0), -1)
        # policy = self.pfc(out)
        # out = self.vl(s)
        # out = out.view(out.size(0), -1)
        # value = self.vfc(out)

        ## structure 3 ##
        pi1 = F.relu(self.pi1(s))
        policies = self.pfc(pi1)
        v1 = F.relu(self.v1(s))
        values = self.vfc(v1)

        return policies, values



    def choose_action(self, s):
        self.eval()
        policy, value = self.forward(s)
        action_prob = F.softmax(policy, dim=-1)
        cat = torch.distributions.Categorical(action_prob)
        action = cat.sample()
        return action

    def loss_func(self, s, a, v_t):
        self.train()
        policies, values = self.forward(s)
        td = v_t - values
        print('td',td)
        c_loss = td.pow(2) # grad_c = td^2

        probs = F.softmax(policies, dim=1)
        m = self.distribution(probs)
        print(m)
        print(td.detach().squeeze())
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v # grad_a = -td^log(a)
        total_loss = (c_loss + a_loss).mean()

        # losslog = open('loss_log.txt', 'a')
        # losslog.flush()
        # losslog.write('v_t is:' + str(v_t) + '\n')
        # losslog.write('policy is:' + str(policies) + '\n')
        # losslog.write('value is:' + str(values) + '\n')
        # losslog.write('td is:' + str(td) + '\n')
        # losslog.write('c_loss is:' + str(c_loss) + '\n')
        # losslog.write('prob is:' + str(probs) + '\n')
        # losslog.write('m is:' + str(m) + '\n')
        # losslog.write('m.log_prob(a) is:' + str(m.log_prob(a)) + '\n')
        # losslog.write('exp_v is:' + str(exp_v) + '\n')
        # losslog.write("a_loss: " + str(a_loss) + '\n')
        # losslog.write("total_loss: " + str(total_loss) + '\n')
        # losslog.close()

        return total_loss

MAX_EPISODE = 200
K_STEPS = 10
env = Environment()
N_S = env.s_dim
GAMMA =  0.9

check_mem = True

nets = []
optims = []

# add agent and optimizer
# for param in single_keys:
#     net = A2CNet(N_S,len(ACTION_BOUND),param)
#     nets.append(net)
#     optim = torch.optim.Adam(net.parameters(), lr=0.01)
#     optims.append(optim)
# for group in group_keys:
#     group_dir = group_items_keys[group]
#     for param in group_dir:
#         net = A2CNet(N_S, len(ACTION_BOUND), param)
#         nets.append(net)
#         optim = torch.optim.Adam(net.parameters(), lr=0.01)
#         optims.append(optim)
#
#
# buffer_a = []
# buffer_r = []
# buffer_s = []
# t1 = time.time()
# s = env.reset()
# for i_episode in range(MAX_EPISODE):
#     t = 0
#
#     t2 = time.time()
#     if t2 - t1 >= 43200.0:  # 12h
#         break
#     while True:
#         t2 = time.time()
#         if t2-t1 >= 43200.0:  # 12h
#             break
#         actions = []
#         for net in nets:
#             s = torch.tensor(s, dtype=torch.float32)
#             a = net.choose_action(s)
#             actions.append(a)
#         s_, r, done, info = env.step(actions)
#         if not done: # generation, restart exploration and get penalty
#             s_ = env.reset()
#             r = -4
#         s_ = torch.tensor(s_, dtype=torch.float32)
#
#         buffer_a.append(actions)
#         buffer_s.append(s_)
#         buffer_r.append(r)
#
#         s = s_
#         t += 1
#
#         if t == K_STEPS:
#             for n in range(len(nets)): # Kstep is over
#                 # calculate the loss function and backpropagate
#                 v_s_ = nets[n](buffer_s[-1])[-1].data.numpy()[0] # V(s_t+n)
#                 buffer_v_target = []
#                 buffer_r = buffer_r[:-1]
#                 for r in buffer_r[::-1]:  # reverse buffer r
#                     v_s_ = r + GAMMA * v_s_
#                     buffer_v_target.append(v_s_)
#
#                 buffer_v_target.reverse()
#
#                 ss = torch.stack(buffer_s[:-1])
#                 a = torch.Tensor(np.array(buffer_a)[:-1, n])
#                 v_t = torch.Tensor(np.array(buffer_v_target)[:, None])  # .unsqueeze(1)
#
#                 loss = nets[n].loss_func(ss, a, v_t)  # gradient = grad[r + gamma * V(s_) - V(s)]
#                 optims[n].zero_grad()
#                 loss.backward()
#                 optims[n].step()
#
#             buffer_a = []
#             buffer_r = []
#             buffer_s = []
#
#             break