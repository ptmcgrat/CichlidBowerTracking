#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:15:16 2023

@author: bshi
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms, models
from torch.nn import functional as F
import cv2 
from PIL import Image
import datetime

class AddFishSexPreparer():
    
    #This class takes in SORT trackfile and video information and performs the following:
    #1.) run a FishSexClassifer Model on each object detection
    #2.) averages the sex_class with respect to SORT tracks
    
    def __init__(self, fileManager, videoIndex):
        self.__version__ = '1.0.0'
        self.fileManager = fileManager
        self.videoObj = self.fileManager.returnVideoObject(videoIndex)
        self.validateInputData()
        
        self.batch_size=50
        self.num_workers=0
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def validateInputData(self):
        
        assert os.path.exists(self.localSexClassificationModelFile)
        assert os.path.exists(self.videoObj.localVideoFile)
        assert os.path.exists(self.fileManager.localTroubleshootingDir)
        assert os.path.exists(self.videoObj.localFishTracksFile)
        return
        
    def RunFishSexClassifier(self):
            
        print('Running Fish Sex Classifier on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)
        dataloaders = {'predict':torch.utils.data.DataLoader(MFDataset(self.videoObj.localFishTracksFile, self.videoObj.localVideoFile),batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers)}
        
        model = models.resnet50(pretrained=False).to(self.device)
        model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True),nn.Linear(128, 2)).to(self.device)
        model.load_state_dict(torch.load(self.localSexClassificationModelFile)) 
        model.eval()
        
        tracks = pd.read_csv(self.videoObj.localFishTracksFile)
        sex_df = pd.DataFrame(columns=list(tracks.columns)+['sex_class', 'sex_p_value'])
        count=0
        
        for i, idx in dataloaders['predict']:
            current_track=tracks.iloc[list(idx), 0: ]
            sample=torch.stack( [j.to(self.device) for j in i])
            pred_logits_tensor=model(sample)
            pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
            pred_class=np.argmax(pred_probs, axis=1)
            pred_acc=np.max(pred_probs, axis=1)
            
            current_track['sex_class']=pred_class
            current_track['sex_p_value']=pred_acc
            sex_df = pd.concat([sex_df, current_track], ignore_index=True)
            
            if count.__mod__(1000)==0:
                print('Proccessing: count'+ str(count*self.batch_size)+ 'of'+ str(len(tracks.frame)))
            count+=1
        
        sex_df.to_csv(self.videoObj.localFishSexFile)

class MFDataset(Dataset):
    
    #This allows us to predict on detections in batches
    
    def __init__(self, csv_file, VideoFile):
        
        self.tracks = pd.read_csv(csv_file)
        self.device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.video=VideoFile
        
    def __len__(self):
        
        return (len(self.tracks.frame))
        
    def __getitem__(self, idx):
        
        current_track=self.tracks.iloc[idx, ]

        cap = cv2.VideoCapture(self.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_track.frame))
        
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret, frame = cap.read()

        xc=(current_track.xc)
        yc=(current_track.yc)
        w=(current_track.w)
        h=(current_track.h)
        delta_xy=(int((1/2)*max(int(w)+1, int(h)+1)))+10
        frame = frame[int(max(0, yc - delta_xy)):int(min(yc + delta_xy,height)) , int(max(0, xc - delta_xy)):int(min(xc + delta_xy, width))]
        frame=cv2.resize(frame, (100, 100))
        
        frame = frame[...,::-1]
        img = Image.fromarray(frame, "RGB")        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        trans=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])
        sample=trans(img)

        return sample, idx
