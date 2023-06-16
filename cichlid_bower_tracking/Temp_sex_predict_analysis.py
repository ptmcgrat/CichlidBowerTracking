    from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, utils, models
from torch.nn import functional as F
import cv2 
from PIL import Image
import pdb
import torch.multiprocessing as mp

import argparse, pdb, datetime
import pandas as pd

from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

class MFDataset(Dataset):
    """dataset takes in a video and gives back folder of fish images to be predicted on"""

    def __init__(self, csv_file, base_name, device):
        """
        Arguments:
            csv_file (string): track file
            root_dir (string): Directory with video
            base_name video number we are analyzing
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tracks = pd.read_csv(csv_file)
        self.base_name=base_name
        self.tracks=self.tracks[self.tracks.base_name==self.base_name]
        self.device =device

    
    def __len__(self):
        return (len(self.tracks.frame))
    
    def __getitem__(self, idx):
        current_track=self.tracks.iloc[idx, ]
        #print(current_track)
        cap = cv2.VideoCapture('./Videos/'+self.base_name+'.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_track.frame))
        
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret, frame = cap.read()
        #get fish
        delta_xy=(int((1/2)*max(int(current_track.w)+1, int(current_track.h)+1)))+10
        frame = frame[int(max(0, current_track.yc - delta_xy)):int(min(current_track.yc + delta_xy,height)) , int(max(0, current_track.xc - delta_xy)):int(min(current_track.xc + delta_xy, width))]
        frame=cv2.resize(frame, (100, 100))
        #transform   
        img = Image.fromarray(frame, "RGB")        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        trans=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])
        sample=trans(img)
        #print(sample)
        return sample, idx


parser = argparse.ArgumentParser(
    description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisID', type = str, help = 'ID of analysis state name')
args = parser.parse_args()

fm_obj = FM(analysisID = args.AnalysisID)
fm_obj.downloadData(fm_obj.localSummaryFile)

dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str,'TrackFish':str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})

# Identify projects to run on:
sub_dt = dt[dt.TrackFish.str.upper() == 'TRUE'] # Only analyze projects that are indicated
projectIDs = list(sub_dt.projectID)




device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TorchModel=fm_obj.localTorchWeightsFile
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True),nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load(TorchModel)) 
model.eval()

for projectID in projectIDs:
    fm_obj.createProjectData(projectID)
    fm_obj.downloadData(fm_obj.localAllFishTracksFile)
    fm_obj.downloadData(fm_obj.localAllFishDetectionsFile)
    fm_obj.downloadData(fm_obj.localOldVideoCropFile)
    #ensure this is the right movie with euthanuization data
    base_name=fm_obj.lp.movies[-1].baseName
    
    #download the video 
    
    csv_file=fm_obj.localAllFishTracksFile
    
    dataloaders = {
        'predict':
        torch.utils.data.DataLoader(MFDataset(csv_file, base_name, device),
                                    batch_size=50, 
                                    shuffle=True,
                                    num_workers=0)}
    tracks = pd.read_csv(csv_file)
    tracks=tracks[tracks.base_name==base_name]
    sex_df = pd.DataFrame(columns=list(tracks.columns)+['sex', 'sex_acc'])
    print('Creating csv for ...'+str(projectID))
    print('processing ...'+base_name)
    count=0
    for i, idx in dataloaders['predict']:
    #predict
        current_track=tracks.iloc[list(idx), 1: ]
        sample=torch.stack( [j.to(device) for j in i])
        pred_logits_tensor=model(sample)
        pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
        pred_class=np.argmax(pred_probs, axis=1)
        pred_acc=np.max(pred_probs, axis=1)
        current_track['sex']=pred_class
        current_track['sex_acc']=pred_acc
        sex_df = pd.concat([ sex_df, current_track], ignore_index=True)
        if count.__mod__(1000)==0
        print(str(count*50)+'of   '+str(len(tracks.index)))
        count+=1
        
    sex_df.to_csv('./sex_df.csv')