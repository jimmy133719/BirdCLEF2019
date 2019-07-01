import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def main ():
    csv_path = 'train_v4.csv'
    f = pd.read_csv(csv_path,header= None)
    data = f.iloc
    Y = data[:,0]
    X = data[:,1:]

    X_res1, X_fold1, y_res1, y_fold1 = train_test_split(X, Y, test_size=0.2, random_state=1,stratify = Y)
    # X_res2, X_fold2, y_res2, y_fold2 = train_test_split(X_res1, y_res1, test_size=0.25, random_state=15,stratify = y_res1)
    # X_res3, X_fold3, y_res3, y_fold3 = train_test_split(X_res2, y_res2, test_size=0.33, random_state=18,stratify = y_res2)
    # X_fold4, X_fold5, y_fold4, y_fold5 = train_test_split(X_res3, y_res3, test_size=0.5, random_state=17,stratify = y_res3)

    fold1 = open('fold1.csv','w+')
    fold2 = open('fold2.csv','w+')
    # fold3 = open('allsplit/fold3.csv','w+')
    # fold4 = open('allsplit/fold4.csv','w+')
    # fold5 = open('allsplit/fold5.csv','w+')
    for i,j in zip ( X_fold1.index , y_fold1):
        buf = str(j) +','+X_fold1[1][i]
        print(buf,file = fold1)
    for i,j in zip ( X_res1.index , y_res1):
        buf = str(j) +','+X_res1[1][i]
        print(buf,file = fold2)
    #for i,j in zip ( X_fold3.index , y_fold3):
    #    buf = str(j) +','+X_fold3[1][i]
    #    print(buf,file = fold3)
    #for i,j in zip ( X_fold4.index , y_fold4):
    #    buf = str(j) +','+X_fold4[1][i]
    #    print(buf,file = fold4)
    #for i,j in zip ( X_fold5.index , y_fold5):
    #    buf = str(j) +','+X_fold5[1][i]
    #    print(buf,file = fold5)
if __name__ =='__main__':
    main()
