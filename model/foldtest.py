from __future__ import print_function
import argparse
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
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
from resnet import BasicBlock , ResNet , Bottleneck , baselinemodel , inception_v3 , Inception3
import time
import pickle
from torch.utils.data import random_split as rsp
import pandas as pd
#from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix
from PIL import Image
import seaborn as sn
import pandas as pd
from sklearn.utils.multiclass import unique_labels
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default = 'folder',help='folder')
parser.add_argument('--dataroot',default = "/media/labhdd/ASAS/train/spec_v2_all/",help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--step', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--saveroot',default = '/media/labhdd/ASAS/kfoldallsplit/',help = 'save path')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.saveroot)
except OSError:
    pass

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def test(args, model, device ,criterion, test_list,f):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                                                                   #
    #   This is the function of testing.                                #
    #   Which evaluate the score of your model.                         #
    #   Input : model , data and the data is store inside device        #
    #   It will evaluate the loss and accuracy of the model             #
    #                                                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    model.eval()
    test_loss = 0
    correct = 0
    test_len = 0.0
    with torch.no_grad():
        for test_loader in test_list:
            test_len += len(test_loader.dataset)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target) # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= test_len/args.batchSize
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_len,
        100. * correct / test_len),file = f)

if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #model = baselinemodel(BasicBlock,[2,2,2,2],660).to(device)
    K = 5
    my_t = transform = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor()
                                    ])
                                
    train_1 = mydataset('allsplit/fold1.csv' , opt.dataroot,my_t)
    train_2 = mydataset('allsplit/fold2.csv' , opt.dataroot,my_t)
    train_3 = mydataset('allsplit/fold3.csv' , opt.dataroot,my_t)
    train_4 = mydataset('allsplit/fold4.csv' , opt.dataroot,my_t)
    train_5 = mydataset('allsplit/fold5.csv' , opt.dataroot,my_t)
    
    fold1 = torch.utils.data.DataLoader(train_1, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
    fold2 = torch.utils.data.DataLoader(train_2, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
    fold3 = torch.utils.data.DataLoader(train_3, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
    fold4 = torch.utils.data.DataLoader(train_4, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
    fold5 = torch.utils.data.DataLoader(train_5, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

    train_loader = [fold1,fold2,fold3,fold4,fold5]
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    model_path = "/media/labhdd/ASAS/kfoldallsplit/"
    #criterion = nn.CrossEntropyLoss()
    model = inception_v3().to(device)

    
    for i in range(9,11):
        pred_= np.zeros((0,1))
        target_ = np.zeros((0,1))
        c = 0
        for j in range(K):
            model_name = model_path + str(j) + str(i*10) + "model.p"
            model.load_state_dict(torch.load(model_name))
            model.eval()
            test_loss = 0
            correct = 0
            test_len = 0.0
            with torch.no_grad():
                for data, target in train_loader[j]:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    #test_loss += criterion(output, target) # sum up batch loss
                    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability    
                    if c == 0 :
                        pred_ = pred.cpu().numpy()
                        target_ = target.cpu().numpy()
                        c+=1
                    else:
                        pred = pred.cpu().numpy()
                        pred_ = np.concatenate((pred_,pred))
                        target = target.cpu().numpy()
                        target_ = np.concatenate((target_,target))
                    print(pred_.shape)
                    print(target_.shape)
        C_matrix = confusion_matrix(pred_,target_)
        #m_namw = "epoch"+str(i*10)+"CMatrix.p"
        #pickle.dump(C_matrix,open(m_namw,"wb"))
        df_cm = pd.DataFrame(C_matrix, range(C_matrix.shape[0]),range(C_matrix.shape[1]))
        plt.subplots(figsize=(320,240))
        sn.set(font_scale=1.4)
        fig = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
        fig = fig.get_figure()
        fig.savefig("/media/labhdd/ASAS/"+"epoch"+str(i*10)+"heatmap.png")
        
                    #correct += pred.eq(target.view_as(pred)).sum().item()
            
            
