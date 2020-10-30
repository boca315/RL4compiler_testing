#encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import time,random, os
import numpy as np
from environment4 import Environment, single_keys, group_keys, group_items_keys, defaultConfig, N_PROGRAMS, seed
# from .environment4 import Environment, single_keys, group_keys, group_items_keys,defaultConfig, N_PROGRAMS, seed


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
        policy = self.pfc(pi1)
        v1 = F.relu(self.v1(s))
        value = self.vfc(v1)

        return policy, value



    def choose_action(self, s):
        self.eval()
        policy, value = self.forward(s)
        action_prob = F.softmax(policy, dim=-1)
        cat = torch.distributions.Categorical(action_prob)
        action = cat.sample()
        return action # [1

    def loss_func(self, s, a, v_t):
        self.train()
        policy, value = self.forward(s)
        td = v_t - value
        # print('td',td.size())
        c_loss = td.pow(2)
        # print('c_loss', c_loss)
        # probs = F.softmax(policy, dim=1)
        prob = F.softmax(policy, dim=0) # 一维时使用dim=0，使用dim=1报错
        # print('prob',prob)
        m = self.distribution(prob)
        # print(a.size(),m)

        # print('m.log_prob(a)',m.log_prob(a).size())
        exp_v = m.log_prob(a) * td.detach().squeeze()
        print(exp_v.size(),c_loss.size())
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

MAX_EPISODE = 200
MAX_EP_STEPS = 5
env = Environment()
N_S = env.s_dim
ACTION_BOUND = [-5,5]
GAMMA = 0.9

check_mem = True

nets = []
optims = []
for param in single_keys:
    net = A2CNet(N_S,len(ACTION_BOUND),param)
    nets.append(net)
    optim = torch.optim.Adam(net.parameters(), lr=0.01)
    optims.append(optim)

for group in group_keys:
    group_dir = group_items_keys[group]
    for param in group_dir:
        net = A2CNet(N_S, len(ACTION_BOUND), param)
        nets.append(net)
        optim = torch.optim.Adam(net.parameters(), lr=0.01)
        optims.append(optim)

t1 = time.time()
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []

    t2 = time.time()
    if t2 - t1 >= 43200.0:  # 12h
        break
    while True:
        t2 = time.time()
        if t2-t1 >= 43200.0:  # 12h
            break
        actions = []
        for net in nets:
            s = torch.tensor(s, dtype=torch.float32)
            # print(s.size())
            a = net.choose_action(s)
            actions.append(a)
        s_, r, done, info = env.step(actions)
        if not done: #timeout
            # continue
            s_ = s, r = -4 # reward of all generation config
        s_ = torch.tensor(s_, dtype=torch.float32)



        for n in range(len(nets)):
            # init_s = torch.tensor(env.dic2vals(defaultConfig()))
            v_s_ = nets[n](s_)[-1].data.numpy()[0] # nets[n].forward(s_)
            print(r ,GAMMA , v_s_)
            v_s_ = r + GAMMA * v_s_
            loss = nets[n].loss_func(s, actions[n], v_s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            optims[n].zero_grad()
            loss.backward()
            optims[n].step()
        s = s_
        t += 1
        ep_rs.append(r)

        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            running_reward = ep_rs_sum
            running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            print("action", actions, "episode:", i_episode, "  reward:", int(running_reward))
            res = open('status.txt', 'a')
            res.write('episode: '+str(i_episode)+',reward: ' +str(int(running_reward)))
            res.flush()
            res.close()
            break