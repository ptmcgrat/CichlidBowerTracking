#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:05:02 2023

@author: bshi
"""
import cv2
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
import clean_csv as cc
import datetime

#import create_video as cv
import os
def createMaleFemaleAnnotationVideos(videoObj, df, delta_xy = 75):
    videopath=fm.localProjectDir+videoObj.mp4_file
    caps = cv2.VideoCapture(videopath)
    maledf=df[df.average_sex_class>0.5]
    femaledf=df[df.average_sex_class<0.5]
    
    male_track_counts=maledf.groupby('track_id')['track_id'].counts().sort_values(ascending=False)
    female_track_counts=femaledf.groupby('track_id')['track_id'].counts().sort_values(ascending=False)
    
    female_10=female_track_counts.head(10)
    male_10=male_track_counts.head(10)
    
    for i in female_10.index:
        outVideoFile = fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Female.mp4'
        tracks = df[ (df.track_id == i)]
        #outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (videoObj.width, videoObj.height))
        outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (2*delta_xy, 2*delta_xy))
        caps.set(cv2.CAP_PROP_POS_FRAMES, tracks.frame.min())
        current_frame = tracks.frame.min()
        for j,current_track in tracks.iterrows():
            ret, frame = caps.read()
            while current_track.frame != current_frame:
                ret, frame = caps.read()
                current_frame += 1

            #cv2.rectangle(frame, (int(current_track.xc - delta_xy), int(current_track.yc - delta_xy)), (int(current_track.xc + delta_xy), int(current_track.yc + delta_xy)), (255,0,0), 2)
            #cv2.rectangle(frame, (int(current_track.yc - delta_xy), int(current_track.xc - delta_xy)), (int(current_track.yc + delta_xy), int(current_track.xc + delta_xy)), (255,0,0), 2)
            #outAll.write(frame)
            outAll.write(frame[int(current_track.yc - delta_xy):int(current_track.yc + delta_xy), int(current_track.xc - delta_xy):int(current_track.xc + delta_xy)])
            current_frame += 1
        outAll.release()
        fm.uploadData(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Female.mp4')
        os.remove(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Female.mp4')
    for i in male_10.index:
        outVideoFile = fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Male.mp4'
        tracks = df[ (df.track_id == i)]
        #outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (videoObj.width, videoObj.height))
        outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (2*delta_xy, 2*delta_xy))
        caps.set(cv2.CAP_PROP_POS_FRAMES, tracks.frame.min())
        current_frame = tracks.frame.min()
        for j,current_track in tracks.iterrows():
            ret, frame = caps.read()
            while current_track.frame != current_frame:
                ret, frame = caps.read()
                current_frame += 1

            #cv2.rectangle(frame, (int(current_track.xc - delta_xy), int(current_track.yc - delta_xy)), (int(current_track.xc + delta_xy), int(current_track.yc + delta_xy)), (255,0,0), 2)
            #cv2.rectangle(frame, (int(current_track.yc - delta_xy), int(current_track.xc - delta_xy)), (int(current_track.yc + delta_xy), int(current_track.xc + delta_xy)), (255,0,0), 2)
            #outAll.write(frame)
            outAll.write(frame[int(current_track.yc - delta_xy):int(current_track.yc + delta_xy), int(current_track.xc - delta_xy):int(current_track.xc + delta_xy)])
            current_frame += 1
        outAll.release()
        fm.uploadData(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Male.mp4')
        os.remove(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.base_name + '__' + str(i) + '__Male.mp4')
    os.remove(videopath)
def get_projects(fm_obj):
    #fm_obj.downloadData(fm_obj.localSummaryFile)
    dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str,'TrackFish':str, 'AssociateClustersWithTracks':str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})

    # Identify projects to run on:
    projectIDs = list(dt.projectID)
    return projectIDs

# Identify projects to run analysis on
fm_obj = FM(analysisID = 'Single_nuc_1')
#fm_obj.downloadData(fm_obj.localSummaryFile)
#fm_obj.downloadData(fm_obj.localEuthData)

summary_file = fm_obj.localSummaryFile # Shorthand to make it easier to read
projectIDs = get_projects(fm_obj)


print('This script will analyze the folllowing projectIDs: ' + ','.join(projectIDs))


# To run analysis efficiently, we download and upload data in the background while the main script runs
uploadProcesses = [] # Keep track of all of the processes still uploading so we don't quit before they finish

dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str, 'TrackFish':str, 'AssociateClustersWithTracks': str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})
de = pd.read_csv(fm_obj.localEuthData, index_col = False)
trialidx={}
for pid in dt.projectID:
    temp_de=de[de.pid==pid]
    pid_et=datetime.datetime.strptime(str(temp_de.dissection_time.values[0]), "%m/%d/%Y %H:%M")
    fm_obj=FM(projectID = pid, analysisID = "Single_nuc_1")
    videos = fm_obj.lp.movies
    count=0
    for videoIndex in videos:
        delta=videoIndex.endTime-pid_et
        days=delta.total_seconds() / (60*60*24)
        if days<1:
            if days>0:
                trialidx[pid]=count
        count+=1

for i in projectIDs:
    fm=FM(analysisID = 'Single_nuc_1', projectID=i)
    #fm.downloadData(fm.localAllTracksAsscociationFile)
    #fm.downloadData(fm.localAllLabeledClustersFile)
    fm.downloadData(fm.localAllFishSexFile)
    
    #fm.downloadData(fm.localAllFishTracksFile)
    #fm.downloadData(fm.localAllFishDetectionsFile)
    
    videoObj=fm.returnVideoObject(trialidx[i])
    fm.downloadData(fm.localProjectDir+videoObj.mp4_file)
    print('done with download for '+str(i))
    csv_dict={'p': fm.localAllFishSexFile}
    df_dict=cc.clean_csv(csv_dict, videoObj.baseName).output()
    createMaleFemaleAnnotationVideos(videoObj, df_dict['p'])
    print('made videos for '+str(i))
