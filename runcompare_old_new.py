import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 


from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM



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

paths=['/home/bshi/Dropbox (GaTech)/BioSci-McGrath/PublicIndividualData/Breanna/__ProjectData/Single_nuc_1/', '/MasterAnalysisFiles/', 'AllDetectionsFish.csv', 'AllTrackedFish.csv']
for i in projectIDs: 
    fm=fm_obj = FM(analysisID = 'Single_nuc_1', projectID=i)
    nt = pd.read_csv(fm.localAllFishTracksFile)
    fm.downloadData(fm.localAllFishDetectionsFile)
    nd = pd.read_csv(fm.localAllFishDetectionsFile)
    
    ns=pd.read_csv(fm.localAllFishSexFile)
    ot=pd.read_csv(paths[0]+i+paths[1]+paths[3])
    od=pd.read_csv(paths[0]+i+paths[1]+paths[2])
    p_annotations=track_annotations[track_annotations.trial==i]
    for idx, row in p_annotations.iterrows():
        dgfc
    


