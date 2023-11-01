import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 


from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

from cichlid_bower_tracking.data_preparers.cluster_sex_association_preparer import ClusterSexAssociationPreparer as CSAP



projectID='MC_singlenuc21_6_Tk53_030320'
AnalysisType='AssociateClustersWithSex'
AnalysisID='Single_nuc_1'
# Identify projects to run analysis on
fm_obj = FM(analysisID = AnalysisID)

summary_file = fm_obj.localSummaryFile # Shorthand to make it easier to read
# To run analysis efficiently, we download and upload data in the background while the main script runs
uploadProcesses = [] # Keep track of all of the processes still uploading so we don't quit before they finish

dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str, 'TrackFish':str, 'AssociateClustersWithTracks': str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})
de = pd.read_csv(fm_obj.localEuthData, index_col = False)
dh = pd.read_csv(fm_obj.localLastHourFrameFile, index_col = False)
trialidx={}
for pid in dt.projectID:
    temp_de=de[de.pid==pid]
    pid_et=datetime.datetime.strptime(str(temp_de.dissection_time.values[0]), "%m/%d/%Y %H:%M")
    fm_obj=FM(projectID = pid, analysisID = AnalysisID)
    videos = fm_obj.lp.movies
    count=0
    for videoIndex in videos:
        delta=videoIndex.endTime-pid_et
        days=delta.total_seconds() / (60*60*24)
        if days<1:
            trialidx[pid]=videoIndex.baseName
        count+=1

dh=pd.DataFrame(columns=list(dh.columns))
for pid in dt.projectID:
    temp_de=de[de.pid==pid]
    pid_et=datetime.datetime.strptime(str(temp_de.dissection_time.values[0]), "%m/%d/%Y %H:%M")
    fm_obj=FM(projectID = pid, analysisID = AnalysisID)
    videos = fm_obj.lp.movies
    for videoIndex in videos:
        delta=videoIndex.endTime-pid_et
        days=delta.total_seconds() / (60*60*24)
        if trialidx[pid]==videoIndex.baseName:
                stime=int((pid_et-datetime.timedelta(hours=1)-videoIndex.startTime).total_seconds()*29)
                etime=int((pid_et-videoIndex.startTime).total_seconds()*29)
                new_row = {'Unnamed: 0': videoIndex.baseName,'trial': pid, 'sframe': stime, 'eframe': etime, 'euth-time': pid_et, 'video_end': videoIndex.endTime, 'total_sec_end_et': days }
                df2 = pd.DataFrame([new_row])
                dh = pd.concat([dh, df2], ignore_index=True)

dh.to_csv(fm_obj.localLastHourFrameFile, index=False)


fm_obj.uploadData(fm_obj.localLastHourFrameFile)

p1=subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data',AnalysisType, '--ProjectID', projectID, '--AnalysisID', AnalysisID])
p1.communicate()
#1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.associate_clusters_with_sex', projectID, AnalysisID])
#1.communicate()
fm_obj = FM(analysisID = AnalysisID, projectID=projectID)
CSAP(fm_obj).runAssociationAnalysis()

