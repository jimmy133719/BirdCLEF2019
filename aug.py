# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:34:19 2019

@author: Jimmykoh
"""
import os, os.path
import cv2
import numpy as np
import config as cfg
from utils import image
import spec
from utils import log

AUDIO_PATH = 'directory of audio'
DATASET_PATH = 'directory to save target spectrogram'
DATASET_NOISE_PATH = 'directory to save noise spectrogram'
min_spec_num = 60
RANDOM = cfg.getRandomState()

def count_spec_num(path):
    spec_num = []
    index = 0
    for subject in os.listdir(path):
        spec_data = os.path.join(path, subject)
        spec_num.append(len([name for name in os.listdir(spec_data) if os.path.isfile(os.path.join(spec_data, name))]))
        if spec_num[index] < min_spec_num:
            print(subject)
        index += 1
    return spec_num

def call_augmentation(spec_num,path):
    i = 0
    for subject in os.listdir(path):
        #if(spec_num[i]<min_spec_num): #origin
        #print(subject)
        spec_path = os.path.join(path, subject)
        spec_noise_path = os.path.join(DATASET_NOISE_PATH, subject)
        if not os.path.exists(spec_noise_path):
          os.makedirs(spec_noise_path)
        #c = 0 #origin
        #while(c < min_spec_num): #origin
        for file in os.listdir(spec_path):
            img = image.openImage(os.path.join(spec_path,file))
            img = image.augment(img, cfg.IM_AUGMENTATION, cfg.AUGMENTATION_COUNT, cfg.AUGMENTATION_PROBABILITY)
            #img_aug_path = spec_path + '/' + str(c) + file #origin
            img_aug_path = spec_noise_path + '/noise_' + file
            cv2.imwrite(img_aug_path, 255*img)
                #c += 1 #origin
        i += 1
        #print(i)

def parseDataset():

    miss = 0
    miss_spec = []
    for i in range(len(os.listdir(AUDIO_PATH))):
        if not os.listdir(AUDIO_PATH)[i] in os.listdir(DATASET_PATH):
            miss_spec.append(os.listdir(AUDIO_PATH)[i])
            miss = miss + 1
    miss_spec = sorted(miss_spec)
    print(miss)
    print(miss_spec) 
    
    # List of classes, subfolders as class names
    #CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'audio')))]

    #switch = 0
    # Parse every class
    for c in miss_spec:
        #if c == 'norfli':
        #   switch = 1
        #if switch == 1:    
        # List all audio files
        afiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'audio', c)))]

        # Calculate maximum specs per file
        max_specs = cfg.MAX_SPECS_PER_CLASS // len(afiles) + 1

        # Get specs for every audio file
        for i in range(len(afiles)):
            
            spec_cnt = 0

            try:
                
                # Stats
                print(i + 1, '/', len(afiles), c, afiles[i])

                # Get specs and signal to noise ratios
                specs, noise = spec.getSpecs(os.path.join(cfg.TRAINSET_PATH, 'audio', c, afiles[i]))
                # print('number of specs = ' + str(specs))
                threshold = 0
                max_index = 0
                # Save specs if it contains signal
                for s in range(len(specs)):
                    # NaN?
                    if np.isnan(noise[s]):
                        noise[s] = 0.0

                    # Above SIGNAL_THRESHOLD?
                    if noise[s] >= threshold:
                        threshold = noise[s]
                        max_index = s
                        #print(threshold)
                        # Create target path for accepted specs
                        spec_filepath = os.path.join(cfg.DATASET_SPECIAL_PATH, c)
                        if not os.path.exists(spec_filepath):
                            os.makedirs(spec_filepath)
                        
                    else:
                        # Create target path for rejected specs -
                        # but we don't want to save every sample (only 10%)
                        if RANDOM.choice([True, False], p=[0.1, 0.90]):
                            filepath = os.path.join(cfg.NOISE_SPECIAL_PATH)
                            if not os.path.exists(filepath):
                                os.makedirs(filepath)
                            if filepath:
                                # Filename contains s2n-ratio
                                filename = str(int(noise[s] * 10000)).zfill(4) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)
        
                                # Write to HDD
                                cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)  
                                                            
                        else:
                            filepath = None
                            
                    if s == (len(specs)-1):                    
                        # Filename contains s2n-ratio
                        filename = str(int(noise[max_index] * 10000)).zfill(4) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)

                        # Write to HDD
                        cv2.imwrite(os.path.join(spec_filepath, filename + '.png'), specs[max_index] * 255.0) 

                        # Count specs
                        spec_cnt += 1

                       

                    # Do we have enough specs already?
                    if spec_cnt >= max_specs:
                        break

                # Stats
                log.i((spec_cnt, 'specs'))       

            except:
                log.e((spec_cnt, 'specs', 'ERROR DURING SPEC EXTRACT'))
                continue


if __name__ == '__main__':
    spec_num = count_spec_num(DATASET_PATH)
    print(len(spec_num))
    print(min(spec_num))
    print(sum(spec_num)/len(spec_num))
    call_augmentation(spec_num, DATASET_PATH)
    #spec_num_new = count_spec_num(DATASET_PATH)
    #parseDataset()    