'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from vgg import *
from resnet import *
from utils import progress_bar
from opt import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume','-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='vgg', help='which models to use')
parser.add_argument('--optim', default='adam', help='which optimizer to use')
parser.add_argument('--lam', type=float, default=0.1, help="reg. param")
parser.add_argument('--max_epochs', type=int, default=300, help="maximum epochs")
parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
parser.add_argument('--seed', type=int, default=1, help="random seed")
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
if args.model == 'vgg':
    net = VGG("VGG16")
elif args.model == 'resnet':
    net = ResNet34()
else:
    pass
net = net.to(device)
if device == 'cuda':
#    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
criterion.to(device)

if args.optim in ['adam', 'subadaml12', 'subadaml23', 'subadaml1']:
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'adaml0':
    l0_penalty = args.lam / 50000
    optimizer = AdamL0(net.parameters(), lr=args.lr, weight_decay=args.wd, l0_penalty=l0_penalty)
elif args.optim == 'adaml12':
    penalty = args.lam / 50000
    optimizer = AdamL12(net.parameters(), lr=args.lr, weight_decay=args.wd, penalty=penalty)
elif args.optim == 'adaml23':
    penalty = args.lam / 50000
    optimizer = AdamL23(net.parameters(), lr=args.lr, weight_decay=args.wd, penalty=penalty)
elif args.optim == 'adaml1':
    penalty = args.lam / 50000
    optimizer = AdamL1(net.parameters(), lr=args.lr, weight_decay=args.wd, l1_penalty=penalty)
elif args.optim == 'proxsgdl1':
    penalty = args.lam / 50000
    optimizer = ProxSGDL1(net.parameters(), lr=args.lr, weight_decay=args.wd, penalty=penalty)
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for param in net.parameters():
    print (param.shape)

if args.optim == 'adaml0':
    print (l0_penalty)
elif args.optim == 'adaml12':
    print (penalty)
elif args.optim == 'adaml23':
    print (penalty)
elif args.optim == 'adaml1':
    print (penalty)
elif args.optim == 'proxsgdl1':
    print (penalty)

# Training
def train(epoch, optim):
    if optim == 'adam':
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #reg_loss = args.lam * net.reg_loss()
            #total_loss = loss + reg_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    elif optim == 'subadaml12':
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            reg_loss = (args.lam / 50000) * net.l12reg_loss()
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    elif optim == 'subadaml1':
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            iter_reg_loss = (args.lam / 50000) * net.l1reg_loss()
            total_loss = loss + iter_reg_loss
            total_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            reg_loss += iter_reg_loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    elif optim == 'subadaml23':
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            reg_loss = (args.lam / 50000) * net.l23reg_loss()
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    else:
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0.0
        reg_loss = 0.0 
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if args.optim == 'adaml12':
                reg_loss += (args.lam / 50000) * net.l12reg_loss().item()
            elif args.optim == 'adaml23':
                reg_loss += (args.lam / 50000) * net.l23reg_loss().item()
            elif args.optim == 'adaml1' or args.optim == 'proxsgdl1':
                reg_loss += (args.lam / 50000) * net.l1reg_loss().item()
            elif args.optim == 'adaml0' or 'proxsgdl0':
                reg_loss += 0.0
            else:
                raise ValueError("args.optim wrong!!")

            optimizer.step()
            
            #total_loss = loss.item() + reg_loss
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100. * correct/total, correct, total))
    
    return train_loss/(batch_idx+1), reg_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    ratio_3, ratio_4 = net.sparsity_level()
    ratio = net.exact_sparsity()
    print ("Sparsity Ratio: {} and {}".format(ratio_3, ratio_4))
    print ("Exact Sparsity Ratio: {}".format(ratio))
    return 100.*correct/total, ratio_3, ratio_4
    """
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    """

for epoch in range(args.max_epochs):
    if epoch >= 150 and epoch < 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
    elif epoch >= 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.01
    else:
        pass
    train(epoch, args.optim)
    test(epoch)
