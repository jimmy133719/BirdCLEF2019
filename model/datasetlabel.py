import os
import string
from sklearn.model_selection import train_test_split 
path = "/media/labhdd/ASAS/train/spec_v4/"
path2 = "/media/labhdd/ASAS/train/spec_v4_noise/"
label_dict = {}
c = 0
label = []
for i in os.listdir(path):
    label.append(i)
    label_dict[i] = c
    c+=1

f = open("train_v4.csv","w+")

for i in label:
    for j in os.listdir(path + i+'/'):
        print(str(label_dict[i]) +','+ 'spec_v4/'+ i+'/'+j)
        print(str(label_dict[i]) +','+ 'spec_v4/'+ i+'/'+j , file = f)
label2 = []
for j in os.listdir(path2):
    label2.append(j)
for i in label2:
    for j in os.listdir(path2 + i + '/'):
        print(str(label_dict[i]) + ',' + 'spec_v4_noise/' + i + '/' + j)
        print(str(label_dict[i]) + ',' + 'spec_v4_noise/' + i + '/' + j , file = f)
        
