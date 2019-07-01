from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from model import BasicBlock , ResNet , Bottleneck , baselinemodel , inception_v3 , Inception3
import time
from torch.utils.data import random_split as rsp
import pandas as pd
#from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataset import Dataset
from PIL import Image
parser = argparse.ArgumentParser()

parser.add_argument('--dataroot',default = "/media/labhdd/ASAS/train/",help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--step', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.001')
parser.add_argument('--momen', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--saveroot',default = '/media/labhdd/ASAS/Resnet34_1/',help = 'save path')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.saveroot)
except OSError:
    pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Basic') != -1:
        pass
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class mydataset(Dataset):
    def __init__(self,csv_path,img_path,transforms = None):
        self.data_info = pd.read_csv(csv_path,header = None)
        self.img_path = img_path
        self.transform = transforms
        self.X_train = np.asarray(self.data_info.iloc[:, 1:])
        self.y_train = np.asarray(self.data_info.iloc[:, 0])

    def __getitem__(self, index):
        image_name = ''
        image_name = self.X_train[index][0] + image_name
        image_name = self.img_path + image_name
        img = Image.open(image_name)
        img_tensor = img
        if self.transform is not None:
            img_tensor = self.transform(img)
        label = self.y_train[index]
        return (img_tensor, label)
    def __len__(self):
        return len(self.data_info.index)

def resnet10(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)

def test(args, model, device ,criterion, test_loader,f):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)/args.batchSize
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),file = f)

def train(args,model,device,train_loader,optimizer,critetion,epoch,f):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = critetion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            #print('Train Epoch: {} Current Fold: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #   100. * batch_idx / len(train_loader), loss.item()),file = f)
            print('loss: {:.6f}'.format(loss.item()))

if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    #cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #######################################################################################
    # No idea what input is and no idea what transform is needed.
    # dataset = dset.ImageFolder(root=opt.dataroot,
    #                            transform = transforms.Compose([
    #                                transforms.Grayscale(),
    #                                transforms.ToTensor()
    #                                ])
    #                            )
    # assert dataset
    # d_len = len(dataset)
    # print(d_len)
    f = open(opt.saveroot+"log.txt","a+")
    # d1 , d2 = rsp(dataset, [60000 , d_len-60000])
    # dataloadera = torch.utils.data.DataLoader(d1, batch_size=opt.batchSize,
    #                                          shuffle=True, num_workers=int(opt.workers))
    # dataloaderb = torch.utils.data.DataLoader(d2, batch_size=opt.batchSize,
    #                                          shuffle=True, num_workers=int(opt.workers))
    #######################################################################################
    #model = resnet10()
    
    my_t = transform = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize( (0.37684,0.37684,0.37684),(0.19487,0.19487,0.19487) )
                                    ])
                                
    train_dataset = mydataset('fold2.csv' , opt.dataroot,my_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))
    val_dataset = mydataset('fold2.csv' , opt.dataroot,my_t)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))
    
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    #model = ResNet( BasicBlock , [1,1,1,1],300).to(device)
    
    critetion = nn.CrossEntropyLoss()
    # model = inception_v3().to(device)
    f_h = open(opt.saveroot + "hyperparameter.txt","w")
    print(opt,file = f_h)
    model = baselinemodel(BasicBlock,[4,4,4,4],660).to(device)
    # torch.save(base.state_dict(),'ini_model.pt')
    # optimizer = optim.Adam(model.parameters(),lr = opt.lr , betas=(opt.beta1, 0.999))
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum = opt.momen)
    model.apply(weights_init)
    for epoch in range(opt.step):
        t1 = time.time()
        train( opt , model , device , train_loader , optimizer , critetion , epoch,f)
        print("train:",file = f)
        test( opt , model , device , critetion , train_loader,f)
        print("validation:",file = f)
        test( opt , model , device , critetion , val_loader,f)
        t2 = time.time()
        print('current epoch:{}'.format(epoch))
        print("Require {:.4f} s an epoch".format(t2-t1))
        if epoch % 5 == 0:
            torch.save(model.state_dict(),opt.saveroot+'{}modelv4.pt'.format(epoch))

