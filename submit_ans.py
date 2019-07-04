import numpy as np
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch
import torch.nn as nn
import argparse
from resnet import inception_v3, ResNet , BasicBlock ,baselinemodel
from operator import itemgetter
from collections import OrderedDict
from collections import Counter

import config as cfg
from utils import audio
from utils import image
from utils import batch_generator as bg
from utils import log 
import json
import csv
from scipy.signal import butter, lfilter, medfilt2d, wiener
import cv2

x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = OrderedDict(sorted(x.items(), key=itemgetter(1)))

class model_prediction():
    def __init__(self,opt,inx):
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        #model = inception_v3().to(self.device)
        #model = baselinemodel(BasicBlock,[2,2,2,2],660).to(self.device)
        model = baselinemodel(BasicBlock,[3, 4, 6, 3],660).to(self.device)
        model_path = opt.model_path + str(inx) + 'modelv4.pt'
        model.load_state_dict(torch.load(model_path))
        print(model)
        self.model = model
        self.Softmax = nn.Softmax(dim=1)
        self.label_csv = open(opt.csv_path,'r')
    def construct_label_dict(self):
        self.label_dict = {}
        for i in self.label_csv:
            key = i.split(',')
            class_name = key[1].split('/')
            self.label_dict[int(key[0])] = class_name[0]
        return self.label_dict
    def get_probability(self,x,opt):
        # data to device?
        self.model.eval()
        with torch.no_grad():
            d_x = torch.from_numpy(x).float()
            d_x = d_x.to(self.device)
            output = self.model(d_x)#confidence
            p = self.Softmax(output)#prob
            if opt.cuda :
                p = p.cpu()
            p = p.numpy()
            return output,p.flatten()
            
def get_power_spectrum(x):
    return abs(np.fft.fft(x)**2)
    
def spectralFlux(spectra, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    spectralFlux = []

    # Compute flux for zeroth spectrum
    flux = 0
    for bin in spectra[0]:
        flux = flux + abs(bin)

    spectralFlux.append(flux)

    # Compute flux for subsequent spectra
    for s in range(1, len(spectra)):
        prevSpectrum = spectra[s - 1]
        spectrum = spectra[s]

        flux = 0
        for bin in range(0, len(spectrum)):
            diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])

            # If rectify is specified, only return positive values
            if rectify and diff < 0:
                diff = 0

            flux = flux + diff

        spectralFlux.append(flux)

    return spectralFlux

def select_frame_criterion(file):
    # Split signal in consecutive chunks with overlap
    sig, rate = audio.openAudioFile(os.path.join(cfg.TESTSET_PATH,file), cfg.SAMPLE_RATE)
    sig_splits = audio.splitSignal(sig, rate, cfg.SPEC_LENGTH, cfg.SPEC_OVERLAP, cfg.SPEC_MINLEN)
    cnt1 = 0
    power_spectrum = []
    for sig in sig_splits:
        power_spectrum.append(get_power_spectrum(sig))
        #print(max(power_spectrum),file = f_power_spec)
        cnt1 += 1
              
    spec_flux = spectralFlux(power_spectrum,rectify=True)
    #print(spec_flux,file = f_flux)
    #print('\n')
    return spec_flux    

def load_median():
    metadataPath = '/media/labhdd/ASAS/val/val/metadata/'
    median = []
    wavefile = []
    metadata_files = [f for f in sorted(os.listdir(metadataPath)) if f[0]!='.']
    for file in metadata_files:
        with open(os.path.join(metadataPath,file), 'r') as f:
            data = json.load(f)
            median.append(data['MediaId'])
            wavefile.append(data['FileName'])
    dic = dict(zip(wavefile,median))
    return dic
    
def getTimestamp(start, end):

    m_s, s_s = divmod(start, 60)
    h_s, m_s = divmod(m_s, 60)
    start = str(h_s).zfill(2) + ":" + str(m_s).zfill(2) + ":" + str(s_s).zfill(2)

    m_e, s_e = divmod(end, 60)
    h_e, m_e = divmod(m_e, 60)
    end = str(h_e).zfill(2) + ":" + str(m_e).zfill(2) + ":" + str(s_e).zfill(2)

    return start + '-' + end    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',default = "",help='path to dataset')
    #parser.add_argument('--model_path',default = "/media/labhdd/ASAS/V3/110modelv3.pt",help='path to model')
    parser.add_argument('--model_path',default = "/media/labhdd/ASAS/Resnet34_2/",help='path to model')
    parser.add_argument('--csv_path',default = "./train_v3.csv",help='path to label csv')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ignore_prob',default = 1e-4 , type = float)
    opt = parser.parse_args()
    print(opt)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # You should read file here and call power spectrum to  #
    # calculate the value and make decision                 #
    wav_files = [f for f in sorted(os.listdir(cfg.TESTSET_PATH)) if f[0]!='.']
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    dirpath = 'resnet_34_2'
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    for inx in range(5,120,5): # using different epochs
        
        model = model_prediction(opt,inx)
        label = model.construct_label_dict()
    
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        filepath = os.path.join(dirpath,'valset_s2n001_resnet34_%3d.csv' % (inx))
        f = open(filepath,'w+')
    
        
        median_dict = load_median()
        
        #result = []
        #predict_class = []
        #predict_prob = []
        submission = []
        SPEC_SIGNAL_THRESHOLD = 0.001
        num = 0
        predict_prob_threshold = 0.0 
        for file in wav_files:
            if num==5:
              break
              
            #spec_flux = select_frame_criterion(file)
            #print(spec_flux,file = f)
            
            # Get specs for file
            cnt2 = 1
            predict_class = []
            predict_prob = []
            predict_s2n = []
            accept_spec_num = 0
            for spec in audio.specsFromFile(os.path.join(cfg.TESTSET_PATH,file),
                                            cfg.SAMPLE_RATE,
                                            cfg.SPEC_LENGTH,
                                            cfg.SPEC_OVERLAP,
                                            cfg.SPEC_MINLEN,
                                            shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                                            fmin=cfg.SPEC_FMIN,
                                            fmax=cfg.SPEC_FMAX,
                                            spec_type=cfg.SPEC_TYPE):
               
                s2n = audio.signal2noise(spec)
                #print(s2n,file = f)
                # Above SIGNAL_THRESHOLD?
                if s2n >= SPEC_SIGNAL_THRESHOLD:
                
                    # Resize spec
                    #spec = image.resize(spec, cfg.IM_SIZE[0], cfg.IM_SIZE[1], mode=cfg.RESIZE_MODE)
        
                    # Normalize spec
                    #spec = image.normalize(spec, cfg.ZERO_CENTERED_NORMALIZATION)
                    # Prepare as input
                    spec = image.prepare(spec)
                    
                    k = np.random.rand(1,1,256,128)
            
                    confidence,prob = model.get_probability(spec,opt)
                    
        #            print(label)
                    probdict = {}
                    for i in range(659):
                        if prob[i] < opt.ignore_prob:
                            pass
                        else:
                            probdict[label[i]] = prob[i]
                    order = sorted(probdict.items(), key=lambda x: x[1],reverse = True)
                    #print(file + '_' + str(c) + ':' + 'confidence = ' + str(confidence)+'\n',file = f1)
                    #print(file + '_' + str(cnt2) + ':' + 'probablity = ' + str(order[0])+'\n',file = f2)
                    #print(order[0][1])
                    
                    if(order[0][1]>predict_prob_threshold): # select all
                        start = cnt2-cnt2%5
                        end = start+5
                        timestamp = getTimestamp(start,end)#type:string
                        print(median_dict[file]+';'+timestamp+';'+order[0][0]+';'+str(1),file = f)
                        submission.append([median_dict[file]+';'+timestamp+';'+order[0][0]+';'+str(1)])
                        accept_spec_num+=1
                       
                           
                cnt2=cnt2+1
            print('number of audio file : '+str(num+1)+', containing '+str(accept_spec_num)+' specs')
            num += 1  
        del model 
          #print('cnt1 == ' + str(cnt1) + ';cnt2 = ' + str(cnt2))
   # with open('submit_test_v2.csv','w',newline = '') as f:
        #writer = csv.writer(f)
        #writer.writerows(submission)        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    




if __name__ == "__main__":
    main()