import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 


from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
import clean_csv as cc


import create_video as cv
import os

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
fm_obj.downloadData(fm_obj.localLastHourFrameFile,)
lh=pd.read_csv(fm_obj.localLastHourFrameFile, index_col = False)

import pandas as pd
import subprocess
import cv2

cmd1 = "rclone lsf "
cmd2="rclone copy "
cmd3="rm -r "
path='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/PatrickTesting/MaleFemale/'
trialpaths='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
trackfile='/MasterAnalysisFiles/AllTrackedFish.csv'
Annotationpath='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
videos='/Videos/'
here=' ./'
classes=['Male', 'Female']
#get names of tracks needed.

femalelist=subprocess.check_output(cmd1+path+classes[1], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
fdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex'])
for i in femalelist:
    fdf.loc[len(fdf.index)]=i[:-4].split('__')+[classes[1]]

malelist=subprocess.check_output(cmd1+path+classes[0], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
mdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex'])
for i in malelist:
    mdf.loc[len(mdf.index)]=i[:-4].split('__')+[classes[0]]

track_annotations = pd.concat([fdf, mdf], axis=0)
projectIDs.remove('MC_singlenuc62_3_Tk65_060220')

df=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex', 'predicted_sex'])
empty_trials=[]
empty_tracks=[]
error_run=[]
#projectIDs=[ 'MC_singlenuc64_1_Tk51_060220' ]
for i in projectIDs:
    #print(trialidx[i])
    project=track_annotations[track_annotations.trial==i]
    
    project=project[project.base_name=='000'+str(trialidx[i]+1)+'_vid']
    #print(project)
    print(i+':  '+str(len(project.track_id)))
    
    fm=FM(analysisID = 'Single_nuc_1', projectID=i)
    fm.downloadData(fm.localAllFishTracksFile)
    #fm.downloadData(fm.localAllLabeledClustersFile)
    fm.downloadData(fm.localAllFishSexFile)
    csv_dict={'p': fm.localAllFishSexFile, 's': fm.localAllFishTracksFile}
    base_name='000'+str(trialidx[i]+1)+'_vid'
    df_dict=cc.clean_csv(csv_dict, base_name).output()
    sdf=df_dict['p']
    pdf=df_dict['s']
    if set(pdf.track_id.unique())-set(sdf.track_id.unique())!=set():
        error_run.append(i)
    #sdf=df_dict['s']
    #print(sdf)
    if sdf.empty:
        #print(i)
        empty_trials.append(i)
        
    #df=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex', 'predicted_sex'])
    for index, row in project.iterrows():
        ssdf=sdf[sdf.track_id==int(row.track_id)]
        #print('Ground Truth: '+ str(row['sex']))
        if ssdf.empty:
            empty_tracks.append(i+'__'+'000'+str(trialidx[i]+1)+'_vid'+'__'+str(row.track_id))
        elif ssdf.average_sex_class.mean()<0.5:
            row['predicted_sex']='Female'
        else:
            row['predicted_sex']='Male'
        df = pd.concat([df, row.to_frame().T])
    

import pandas as pd
import matplotlib.pyplot as plt

df['idx']=df['trial'].apply(lambda x: '_'.join(x.split('singlenuc')[1].split('_')[:2]))
grouped = df.groupby(['idx'])['acc'].mean()
track_annotations['idx']=track_annotations['trial'].apply(lambda x: '_'.join(x.split('singlenuc')[1].split('_')[:2]))
grouped1 = track_annotations.groupby(['idx'])['sex'].count()
#mcounts = grouped.sex

fig, ax = plt.subplots(figsize=(20,5))
ax.bar(grouped.index, grouped)

ax.set_xlabel('Trial')  
ax.set_ylabel('accuracy of predicted sex to annotated sex')
ax.set_title('Accuracy by Trial')

plt.show()




import numpy as np
means = df.groupby(['idx', 'sex'])['acc'].mean()
df['acc']=df['sex']==df['predicted_sex']

male_counts = means.xs('Male', level=1)
female_counts = means.xs('Female', level=1)
for i in set(female_counts.index)-set(male_counts.index):
    male_counts[i]=-1
for i in set(male_counts.index)-set(female_counts.index):
    female_counts[i]=-1
male_counts=male_counts.sort_index()
female_counts=female_counts.sort_index()
index = np.arange(len(female_counts))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(20,5))

rects1 = ax.bar(male_counts.index, male_counts,bar_width, label='Male')
rects2 = ax.bar(index+bar_width, female_counts,bar_width, label='Female')
ax.set_title('accuracy by Trial')
ax.legend()
plt.show()