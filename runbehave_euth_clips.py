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
fm_obj.downloadData(fm_obj.local10030FrameFile,)
lh=pd.read_csv(fm_obj.local10030FrameFile, index_col = False)
#projectIDs=[ 'MC_singlenuc62_3_Tk65_060220' ]

for i in projectIDs:
    fm=FM(analysisID = 'Single_nuc_1', projectID=i)
    fm.downloadData(fm.localAllTracksAsscociationFile)
    fm.downloadData(fm.localAllLabeledClustersFile)
    fm.downloadData(fm.localAllFishSexFile)
    
    #fm.downloadData(fm.localAllFishTracksFile)
    #fm.downloadData(fm.localAllFishDetectionsFile)
    
    videoobj=fm.returnVideoObject(trialidx[i])
    fm.downloadData(fm.localProjectDir+videoobj.mp4_file)
    print('done with download for '+str(i))
    videopath=fm.localProjectDir+videoobj.mp4_file
    csv_dict={'p': fm.localAllFishSexFile, 'c': fm.localAllLabeledClustersFile, 'a': fm.localAllTracksAsscociationFile}
    #tdf=pd.read_csv(fm.localAllFishTracksFile)
    
    title='euth30100test'
    base_name=videoobj.baseName
    
    fstart=int(lh[lh.trial==i]['sframe'])
    fend=int(lh[lh.trial==i]['eframe'])

    #tdf=tdf[tdf.base_name==base_name]
    #print(fstart>tdf.frame.max())
    
    df_dict=cc.clean_csv(csv_dict, base_name).output()
    #df_dict['p']=df_dict['p'][df_dict['p'].track_id==1741]
    #fstart=df_dict['p'].frame.min()
    #fend=df_dict['p'].frame.max()
    cv.create_video(videopath, df_dict, fstart, fend).run_one_visual(title)
    fm.uploadData(videopath)
    os.remove(videopath)
    print('made video for '+str(i))
    os.remove(fm.localProjectDir+videoobj.mp4_file)
    


