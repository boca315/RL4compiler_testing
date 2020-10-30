#encoding: utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from .data_processer import DataProcessor, batch_size, iftest
from data_processer import DataProcessor, batch_size, iftest
import time


class Net(nn.Module):
    def __init__(self, s_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(s_dim, 50, kernel_size=1, stride=2),
            nn.BatchNorm1d(50),
            nn.ReLU())
            # nn.MaxPool1d(8))
        self.layer2 = nn.Sequential(
            # nn.Conv1d(100, 50, 2),
            nn.Conv1d(50, 100, kernel_size=1),
            nn.BatchNorm1d(100),
            nn.ReLU())
            # nn.MaxPool1d(8))
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        # return F.relu(self.conv1(x))
        # input.shape:(32, 110, 1)
        out = self.layer1(x) # 32, 100, 1
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # 32, 100
        out = self.fc(out)
        return out

logfile = open('logfile', 'a')

logfile.write('begin time: '+str(time.time())+'\n')

data = DataProcessor("/train_dir")
dataset= data.get_torch_dataset()

s_dim = 110
model = Net(s_dim)
loss_fn = nn.NLLLoss()
lr = 1e-2
momentum = 0.5
# optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
weight_decay = 0.01
# optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum, weight_decay=weight_decay) # L2 regularization
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay) # L2 regularization
best_accuracy = 0
best_model = None
n_epochs = 3 if iftest else 50

for epoch in range(1, n_epochs+1):
    train_loader, validation_loader = data.split(dataset)
    train_dataset_len = 0
    vld_dataset_len = 0
    train_num_correct = 0
    vld_num_correct = 0

    logfile.write('---------- Epoch '+str(epoch)+ ' time: ' + str(time.time()) + ' ----------\n')
    logfile.write('\n---------- Train  ----------\n')

    model.train()
    for step, (data, target) in enumerate(train_loader):
        if len(target) != batch_size:
            break
        train_dataset_len += len(target)
        data = data.unsqueeze(2)
        target = target.unsqueeze(1)
        data, target = Variable(data), Variable(target)
        pred = model(data)

        train_loss = loss_fn(pred.unsqueeze(2), target)

        pred = torch.max(pred, 1)[1]
        target = torch.tensor([item[0] for item in target.tolist()])

        train_num_correct += torch.eq(pred, target).sum().float().item()
        train_cur_accuracy = train_num_correct/train_dataset_len

        print("Train Epoch: {}\t step: {}\t Loss: {:.6f}\t Acc: {:.6f} \tlen: {}".format(epoch, step, train_loss.data/train_dataset_len, train_cur_accuracy, train_dataset_len))
        logfile.write('step: '+str(step)+ '\t Loss: '+str(train_loss.data/train_dataset_len)
                         +'\t Train Acc: '+str(train_cur_accuracy) + '\t corr: ' + str(train_num_correct) +'\t len: '+str(train_dataset_len)+'\n')
        logfile.write('pred:  ')
        for i in pred:
            logfile.write(str(int(i.item()))+',')
        logfile.write('\ntarget: ')
        for i in target:
            logfile.write(str(int(i.item()))+',')
        logfile.write('\n')
        logfile.flush()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    logfile.write('\n---------- Validation  ----------\n')
    model.eval()
    for step, (data, target) in enumerate(validation_loader):
        if len(target) != batch_size:
            break
        vld_dataset_len += len(target)

        data = data.unsqueeze(2)
        target = target.unsqueeze(1)
        data, target = Variable(data), Variable(target)
        pred = model(data)
        vld_loss = loss_fn(pred.unsqueeze(2), target)

        pred = torch.max(pred, 1)[1]
        target = torch.tensor([item[0] for item in target.tolist()])

        vld_num_correct += torch.eq(pred, target).sum().float().item()
        vld_cur_accuracy = vld_num_correct / vld_dataset_len
        # print(vld_num_correct)
        print("Valid Acc: {:.6f}\t len:{}".format(vld_num_correct / vld_dataset_len, vld_dataset_len))
        logfile.write('Valid Acc: ' + str(vld_cur_accuracy) + '\t corr: ' + str(vld_num_correct)+ '\t len: ' + str(vld_dataset_len)+ '\n')

        print("Validation Epoch: {}\t step: {}\t Loss: {:.6f}\t Acc: {:.6f} \tlen: {}".format(epoch, step,
                                                                                         vld_loss.data / vld_dataset_len,
                                                                                         vld_cur_accuracy, vld_dataset_len))
        logfile.write('step: ' + str(step) + '\t Loss: ' + str(vld_loss.data / vld_dataset_len)
                         + '\t Acc: ' + str(vld_cur_accuracy) + '\t Best Valid Acc: ' + str(best_accuracy)
                         + '\t corr: ' + str(vld_num_correct) + '\t len: ' + str(vld_dataset_len) + '\n')

        logfile.write('pred:  ')
        for i in pred:
            logfile.write(str(int(i.item())) + ',')
        logfile.write('\ntarget: ')
        for i in target:
            logfile.write(str(int(i.item())) + ',')
        logfile.write('\n')
        logfile.flush()

        if vld_cur_accuracy > best_accuracy:
            best_accuracy = vld_cur_accuracy
            best_model = model

logfile.write('end time: '+str(time.time())+'\n')
torch.save(best_model.state_dict(), 'net.pkl')
# model = Net(s_dim, batch_size)
# model.load_state_dict(torch.load('net.pkl'))

logfile.close()