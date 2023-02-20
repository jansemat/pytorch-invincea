import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import sys
import numpy as np

# config variables
MAL_PATH = "./data/mal.urls" # place malicious URLs here
BENIGN_PATH = "./data/benign.urls" # place benign URLs here
BATCH_SIZE = 256 # make sure this is divisible by 2
CONV_OUT = 256
LINEAR1, LINEAR2 = 1024, 1024
NUM_EPOCH = 50
TOTAL_SIZE = 1000000
TRAIN_SIZE = 900000
TEST_SIZE = 100000
LETTER_VEC_SIZE, DOMAIN_SIZE = 32, 150
NUM_TRAIN_BATCH = int(np.floor(TRAIN_SIZE/BATCH_SIZE))
NUM_TEST_BATCH = int(np.floor(TEST_SIZE/BATCH_SIZE))
OUTFILE="final_results.txt"
# changes:
# - batch size: 64, 128, 256, 512, 1024
# - CONV_OUT: 32, 64, 128, 256, 512
# - LINEAR1/2: 256, 512, 1024
# - LETTER_VEC_SIZE: 16, 32, 64, 128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(39, LETTER_VEC_SIZE, padding_idx=0)
        self.pad1, self.pad2, self.pad3 = nn.ConstantPad1d((0,1),0), nn.ConstantPad1d((0,2),0), nn.ConstantPad1d((0,3),0)
        self.drop = nn.Dropout(p=0.5)
        self.Lnorm1, self.Lnorm2 = nn.LayerNorm([CONV_OUT, LETTER_VEC_SIZE-1]), nn.LayerNorm([CONV_OUT*4])
        self.conv1a = nn.Conv1d(DOMAIN_SIZE,CONV_OUT,2)
        self.conv1b = nn.Conv1d(DOMAIN_SIZE,CONV_OUT,3)
        self.conv1c = nn.Conv1d(DOMAIN_SIZE,CONV_OUT,4)
        self.conv1d = nn.Conv1d(DOMAIN_SIZE,CONV_OUT,5)
        self.fc1 = nn.Linear(CONV_OUT*4, LINEAR1)
        self.fc2 = nn.Linear(LINEAR1, LINEAR2)
        self.fc3 = nn.Linear(LINEAR2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(self.pad1(x)))
        x3 = F.relu(self.conv1c(self.pad2(x)))
        x4 = F.relu(self.conv1d(self.pad3(x)))

        x1, x2, x3, x4 = self.Lnorm1(x1), self.Lnorm1(x2), self.Lnorm1(x3), self.Lnorm1(x4)
        x1 = torch.sum(x1.view(x1.size(0), x1.size(1), -1), dim=2)
        x2 = torch.sum(x2.view(x2.size(0), x2.size(1), -1), dim=2)
        x3 = torch.sum(x3.view(x3.size(0), x3.size(1), -1), dim=2)
        x4 = torch.sum(x4.view(x4.size(0), x4.size(1), -1), dim=2)
        x = torch.cat((self.drop(x1), self.drop(x2), self.drop(x3), self.drop(x4)),1)

        x = self.Lnorm2(x)
        x = self.drop(self.Lnorm2(F.relu(self.fc1(x))))
        x = self.drop(self.Lnorm2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

class Dataset():
    def __init__(self):
        with open(MAL_PATH, "r") as fd:
            self.mal = [(d, 1) for d in np.random.choice(fd.read().splitlines(), size=TOTAL_SIZE, replace=False)]
        with open(BENIGN_PATH, "r") as fd:
            self.ben = [(d,0) for d in np.random.choice(fd.read().splitlines(), size=TOTAL_SIZE, replace=False)]
        self.train_mal, self.train_ben = self.mal[:TRAIN_SIZE], self.ben[:TRAIN_SIZE]
        self.test_mal, self.test_ben = self.mal[-TEST_SIZE:], self.ben[-TEST_SIZE:]
        self.train_mal_loader = torch.utils.data.DataLoader(self.train_mal, batch_size=BATCH_SIZE//2, shuffle=False, num_workers=1)
        self.train_ben_loader = torch.utils.data.DataLoader(self.train_ben, batch_size=BATCH_SIZE//2, shuffle=False, num_workers=1)
        self.test_mal_loader = torch.utils.data.DataLoader(self.test_mal, batch_size=BATCH_SIZE//2, shuffle=False, num_workers=1)
        self.test_ben_loader = torch.utils.data.DataLoader(self.test_ben, batch_size=BATCH_SIZE//2, shuffle=False, num_workers=1)
        self.train_mal_it, self.train_ben_it = iter(self.train_mal_loader), iter(self.train_ben_loader)
        self.test_mal_it, self.test_ben_it = iter(self.test_mal_loader), iter(self.test_ben_loader)

    def domain2intarr(self, domain):
        fqdn = "*abcdefghijklmnopqrstuvwxyz0123456789.-"
        intarr = [fqdn.index(c) for c in domain] + [0]*(DOMAIN_SIZE-len(domain))
        return intarr

    def reset_loaders(self):
        self.train_mal_it, self.train_ben_it = iter(self.train_mal_loader), iter(self.train_ben_loader)
        self.test_mal_it, self.test_ben_it = iter(self.test_mal_loader), iter(self.test_ben_loader)
        return

    def iter_train(self):
        mal_data, mal_labels = next(self.train_mal_it)
        ben_data, ben_labels = next(self.train_ben_it)
        pre_data, pre_labels = (mal_data + ben_data), torch.cat((mal_labels, ben_labels),0)
        ridx = [i for i in range(BATCH_SIZE)]
        np.random.shuffle(ridx)
        ret_domains, ret_labels = [], []

        for i in ridx:
            ret_domains.append(self.domain2intarr(pre_data[i]))
            ret_labels.append(int(pre_labels[i]))
        ret_data = (torch.tensor(ret_domains), torch.tensor(ret_labels))
        return ret_data

    def iter_test(self):
        mal_data, mal_labels = next(self.test_mal_it)
        ben_data, ben_labels = next(self.test_ben_it)
        pre_data, pre_labels = (mal_data + ben_data), torch.cat((mal_labels, ben_labels),0)
        ridx = [i for i in range(BATCH_SIZE)]
        np.random.shuffle(ridx)
        ret_domains, ret_labels = [], []

        for i in ridx:
            ret_domains.append(self.domain2intarr(pre_data[i]))
            ret_labels.append(int(pre_labels[i]))
        ret_data = (torch.tensor(ret_domains), torch.tensor(ret_labels))
        return ret_data


def log2file(epoch,to_log):
    out_log = str(epoch)+"," + ','.join([str(i) for i in to_log]) + "\n"
    with open(OUTFILE, "a+") as fd:
        _ = fd.write(out_log)
    return

def eval_model(net, datasets):
    train_correct, train_total, train_total_loss = 0,0,0
    test_correct, test_total, test_total_loss = 0,0,0
    train_fp, train_fn, test_fp, test_fn = 0,0,0,0 # true/false positive/negative
    crit = nn.BCEWithLogitsLoss(reduction='mean')
    net.eval()

    datasets.reset_loaders()
    for batch_num in range(NUM_TRAIN_BATCH):
        inputs, labels = datasets.iter_train()
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = net(inputs)
        outputs = (outputs.T)[0]
        pred = torch.logical_not(torch.lt(outputs.data, 0))
        train_total += labels.size(0)
        train_correct += (pred == labels.data).sum()
        train_loss = crit(outputs, labels.float())
        train_total_loss += train_loss.item()
        train_fp += torch.logical_and((pred != labels),(pred == True)).sum()
        train_fn += torch.logical_and((pred != labels),(pred == False)).sum()

    for batch_num in range(NUM_TEST_BATCH):
        inputs, labels = datasets.iter_test()
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = net(inputs)
        outputs = (outputs.T)[0]
        pred = torch.logical_not(torch.lt(outputs.data, 0))
        test_total += labels.size(0)
        test_correct += (pred == labels.data).sum()
        test_loss = crit(outputs, labels.float())
        test_total_loss += test_loss.item()
        test_fp += torch.logical_and((pred != labels),(pred == True)).sum()
        test_fn += torch.logical_and((pred != labels),(pred == False)).sum()

    train_fp, train_fn = (train_fp/train_total).item(), (train_fn/train_total).item()
    test_fp, test_fn = (test_fp/test_total).item(), (test_fn/test_total).item()
    trainL, trainA = train_total_loss/train_total, (train_correct.float()/train_total).item()
    testL, testA = test_total_loss/test_total, (test_correct.float()/test_total).item()
    return (trainL, trainA, testL, testA, train_fp, train_fn, test_fp, test_fn)





def main():
    # Build model
    print("Building model...")
    net = Net().cuda()
    net.train()

    # writer = SummaryWriter(log_dir='./log')
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(net.parameters(), lr=0.01)

    with open(OUTFILE, "a+") as fd:
        _ = fd.write("epoch,train_loss,train_acc,test_loss,test_acc,train_fp,train_fn,test_fp,test_fn\n")

    print("Starting training...")
    for epoch in range(NUM_EPOCH):
        # build training/testing datasets
        datasets = Dataset()
        running_loss = 0.0
        for batch_num in range(NUM_TRAIN_BATCH):
            # get training data
            inputs, labels = datasets.iter_train()
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            out = net(inputs)
            out = (out.T)[0]
            loss = crit(out, labels.float())
            loss.backward()
            opt.step()

            # print stats
            # print()
            running_loss += loss.item()
            checkpoint = NUM_TRAIN_BATCH//4
            if batch_num % checkpoint == checkpoint-1:
                print("\tStep: %5d/%5d avg_batch_loss: %.5f" % (batch_num+1, NUM_TRAIN_BATCH, running_loss/checkpoint))
                running_loss = 0.0

        print("\tFinish training this EPOCH, start evaluating...")

        out_tuple = eval_model(net, datasets)
        trainL, trainA, testL, testA = out_tuple[:4]
        log2file(epoch, out_tuple)

        net.train()
        print("EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f" %
                (epoch+1, trainL, trainA, testL, testA))

        del datasets


if __name__ == "__main__":
    main()
