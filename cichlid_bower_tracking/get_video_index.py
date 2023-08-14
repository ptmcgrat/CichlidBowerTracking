#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 00:04:55 2023

@author: bshi
"""

import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 


from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser(
    description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type=str, choices=['Prep', 'Depth', 'Cluster', 'ClusterClassification', 'TrackFish', 'AssociateClustersWithTracks', 'AddFishSex','Summary'],
                    help='Type of analysis to run')
parser.add_argument('AnalysisID', type = str, help = 'ID of analysis state name')
parser.add_argument('--ProjectIDs', type=str, nargs='+', help='Name of projectIDs to restrict analysis to')
parser.add_argument('--Workers', type=int, help='Number of workers')
parser.add_argument('--ModelID', type=str, help='ModelID to use to classify clusters with')
args = parser.parse_args()

def get_projects(fm_obj, analysis_type, fil_projectIDs):
    fm_obj.downloadData(fm_obj.localSummaryFile)
    dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str,'TrackFish':str, 'AssociateClustersWithTracks':str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})

    # Identify projects to run on:
    sub_dt = dt[dt.RunAnalysis.astype(str).str.upper() == 'TRUE'] # Only analyze projects that are indicated
    if analysis_type == 'Prep':
        sub_dt = sub_dt[sub_dt.StartingFiles.astype(str).str.upper() == 'TRUE'] # Only analyze projects that have the right starting files
    elif args.AnalysisType == 'Depth':
        sub_dt = sub_dt[sub_dt.Prep.astype(str).str.upper() == 'TRUE'] # Only analyze projects that have been prepped
    projectIDs = list(sub_dt[sub_dt[analysis_type].astype(str).str.upper() == 'FALSE'].projectID) # Only run analysis on projects that need it

    # Filter out projects if optional argment given
    if fil_projectIDs is not None:
        for projectID in projectIDs:
            if projectID not in fil_projectIDs:
                projectIDs.remove(projectID)
    return projectIDs

# Identify projects to run analysis on
fm_obj = FM(analysisID = args.AnalysisID)
fm_obj.downloadData(fm_obj.localSummaryFile)
fm_obj.downloadData(fm_obj.localEuthData)
if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
    print('Cant find ' + fm_obj.localSummaryFile)
    sys.exit()

summary_file = fm_obj.localSummaryFile # Shorthand to make it easier to read
projectIDs = get_projects(fm_obj, args.AnalysisType, args.ProjectIDs)

if len(projectIDs) == 0:
    print('No projects to analyze')
    sys.exit()

print('This script will analyze the folllowing projectIDs: ' + ','.join(projectIDs))

# Set workers
if args.Workers is None:
    workers = os.cpu_count()
else:
    workers = args.Workers

# To run analysis efficiently, we download and upload data in the background while the main script runs
uploadProcesses = [] # Keep track of all of the processes still uploading so we don't quit before they finish

dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str, 'TrackFish':str, 'AssociateClustersWithTracks': str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})
dt.loc[dt.projectID == projectIDs[0],args.AnalysisType] = 'Running'
dt.to_csv(summary_file, index = False)
fm_obj.uploadData(summary_file)
de = pd.read_csv(fm_obj.localEuthData, index_col = False)
for pid in dt.projectID:
    temp_de=de[de.pid==pid]
    pid_et=datetime.strptime(str(temp_de.dissection_time), "%m/%d/%Y %H:%M")
    print(pid_et)
    print('break')
    fm_obj=FM(projectID = pid, analysisID = args.AnalysisID)
    pdb.set_trace()
    videos = fm_obj.lp.movies
    count=0
    for videoIndex in videos:
        print(videoIndex.endTime)
        print(videoIndex.endTime-pid_et)
        count+=1
    print('cut')
