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
import pdb
#import create_video as cv
import os
import subprocess
import pandas as pd

cmd1 = "rclone lsf "
cmd2="rclone copy "
cmd3="rm -r "
path='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/AddFishSex/'
trialpaths='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
trackfile='/MasterAnalysisFiles/AllTrackedFish.csv'
Annotationpath='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
videos='/Videos/'
here=' ./'
classes=['Male', 'Female','Other']

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
def make_vid(videopath, tracks, outfile, delta_xy=75):
    cap = cv2.VideoCapture(videopath)
    out = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 29, (2*delta_xy, 2*delta_xy))
    cap.set(cv2.CAP_PROP_POS_FRAMES, tracks.frame.min())
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = tracks.frame.min()
    for j1,current_track in tracks.iterrows():
        ret, frame = cap.read()
        while current_track.frame != current_frame:
            ret, frame = cap.read()
            current_frame += 1
        if current_frame!=length:
            #print(length)
            #print(current_frame)
            out.write(frame[max(0, int(current_track.yc - delta_xy)):min(int(current_track.yc + delta_xy),videoObj.height) , max(0, int(current_track.xc - delta_xy)):min(int(current_track.xc + delta_xy), videoObj.width)])
        current_frame += 1
        
    cap.release()
    out.release()
otherlist=subprocess.check_output(cmd1+path+classes[2], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
odf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'predict_sex'])
for i in otherlist:
    odf.loc[len(odf.index)]=i[:-4].split('__')

femalelist=subprocess.check_output(cmd1+path+classes[1], shell=True).decode("utf-8").split('\n')[:-1]
malelist=subprocess.check_output(cmd1+path+classes[0], shell=True).decode("utf-8").split('\n')[:-1]

uni_list=otherlist+femalelist+malelist

for i in list(odf.trial.unique())[1:]:
    print(i)
    fm=FM(analysisID = 'Single_nuc_1', projectID=i)
    #fm.downloadData(fm.localAllTracksAsscociationFile)
    #fm.downloadData(fm.localAllLabeledClustersFile)
    fm.downloadData(fm.localAllFishSexFile)
    
    fm.downloadData(fm.localAllFishTracksFile)
    #fm.downloadData(fm.localAllFishDetectionsFile)
    
    videoObj=fm.returnVideoObject(trialidx[i])
    print('starting download')
    fm.downloadData(fm.localProjectDir+videoObj.mp4_file)
    print('done with download for '+str(i))
    csv_dict={'p': fm.localAllFishSexFile, 's': fm.localAllFishTracksFile}
    df_dict=cc.clean_csv(csv_dict, videoObj.baseName).output()
    df=df_dict['p']
    delta_xy = 75
    videopath=fm.localProjectDir+videoObj.mp4_file
    maledf=df[df.average_sex_class>0.5]
    femaledf=df[df.average_sex_class<0.5]
    add_df=odf[odf.trial==i]
    
    m_add=add_df[add_df.predict_sex=='Male'].shape[0]
    f_add=add_df[add_df.predict_sex=='Female'].shape[0]
    
    male_track_counts=maledf.groupby('track_id')['track_id'].count().sort_values(ascending=False)
    female_track_counts=femaledf.groupby('track_id')['track_id'].count().sort_values(ascending=False)
    female_num=10+int(f_add)
    male_num=10+int(m_add)
    if (len(female_track_counts)<=female_num) or len(male_track_counts)<=male_num:
        missingf=female_num-len(female_track_counts)
        missingm=male_num-len(male_track_counts)
        if missingf>0:
            male_num+=missingf
            female_num-=missingf
        elif missingm>0:
            female_num+=missingm
            male_num-=missingm
    print('Males: '+str(male_num))
    print('Females: '+str(female_num))
    
    female_10=female_track_counts.head(female_num)
    male_10=male_track_counts.head(male_num)
    
    for j in female_10.index:
        name=fm.projectID + '__' + videoObj.baseName + '__' + str(j) + '__Female.mp4'
        outVideoFile = fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(j) + '__Female.mp4'
        if name not in uni_list:
            tracks = df[ (df.track_id == j)]
            make_vid(videopath, tracks, outVideoFile)
            fm.uploadData(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(j) + '__Female.mp4')
            os.remove(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(j) + '__Female.mp4')
        
    for k in male_10.index:
        name= fm.projectID + '__' + videoObj.baseName + '__' + str(k) + '__Male.mp4'
        outVideoFile = fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(k) + '__Male.mp4'
        if name not in uni_list:
            tracks = df[ (df.track_id == k)]
            make_vid(videopath, tracks, outVideoFile)
            fm.uploadData(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(k) + '__Male.mp4')
            os.remove(fm.localAddFishSexDir + fm.projectID + '__' + videoObj.baseName + '__' + str(k) + '__Male.mp4')
    os.remove(videopath)
    print('made videos for '+str(i))


