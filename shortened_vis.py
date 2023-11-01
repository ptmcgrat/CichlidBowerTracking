#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:21:41 2022

@author: bshi42
"""
#testing script
import clean_csv as cc
#import visual_cluster as vc

import create_video as cv
#import line_df as ld

#the purpose of this script is to create video with mutiple types of detections such as:
#yolo detections , sort detections, pytorch class detections, CNN behavior detections. 
# this script can also visualize connections between detections and behaviors. 

#This script can process one video from one trial at a time

# each of these detections is detections are contained in separate csv files 
#here we define all the necessary csv files 


#sortpath='./MasterAnalysisFiles/AllTrackedFish_old.csv'
#this file contains the sort detections
#clusterpath='./MasterAnalysisFiles/AllLabeledClusters.csv'
# this file contains the CNN Behavioral detections
#yolopath='./MasterAnalysisFiles/AllDetectionsFish.csv'
# path to yolo detections
#sumpath='./MasterAnalysisFiles/AllSummarizedTracks.csv'
#path to information about summarized tracks

assoc_path='./MasterAnalysisFiles/AllAssociatedTracks.csv'

pytorch_path='./MasterAnalysisFiles/AllFishSex.csv'

track_path='./MasterAnalysisFiles/AllTrackedFish.csv'
track_path='~/Downloads/AllTrackedFish_5.csv'

base_name='0001_vid'
#this is the 'base_name ' this is a shorten form of the video name
time=15
#time is used in the long video generation method to define how many
# consequetive minutes the video output should be
videopath='./Videos/0001_vid.mp4'
#path to video

#csv_dict={'c': clusterpath, 'y': yolopath, '+': sumpath, 'a': assoc_path, 'p': pytorch_path}

csv_dict={'p': pytorch_path, 'a': assoc_path}

title='addfishsex2'


df_dict=cc.clean_csv(csv_dict, base_name).output()

#add lines

#long cluster
#df_dict['c'], fstart, fend =cv.cut_video().r_behavior(df_dict['c'])
#if 'a' in df_dict.keys():
    #df_dict['a']=df_dict['a'][df_dict['a']['behavior_id'].isin(list(df_dict['c']['behavior_id']))]

fstart, fend =cv.cut_video().r_nothing(30, 45)
#IMG_W = 1296
#IMG_H = 972
#temp fix for old data
#df_dict['s']['y1']=IMG_H*df_dict['s']['y1']
#df_dict['s']['y2']=IMG_H*df_dict['s']['y2']
#df_dict['s']['x1']=IMG_W*df_dict['s']['x1']
#df_dict['s']['x2']=IMG_W*df_dict['s']['x2']
#df_dict['a']['xc']=IMG_W*df_dict['a']['xc']
#df_dict['a']['yc']=IMG_H*df_dict['a']['yc']

#line news to be filtered with respect to clusters as well
fstart=535361
fend=639761
#run video

#last updates to code  
cv.create_video(videopath, df_dict, fstart, fend).run_one_visual(title)
#there is an issue with linedf
#too many lines generated





