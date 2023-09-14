import argparse, pdb, datetime
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM


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


for projectID in projectIDs:
    fm_obj.createProjectData(projectID)
    fm_obj.downloadData(fm_obj.localAllFishTracksFile)
    fm_obj.downloadData(fm_obj.localAllFishDetectionsFile)
    fm_obj.downloadData(fm_obj.localOldVideoCropFile)

    video_crop = np.load(fm_obj.localOldVideoCropFile)
    poly = Polygon(video_crop)

    t_dt = pd.read_csv(fm_obj.localAllFishTracksFile, index_col = 0)
    d_dt = pd.read_csv(fm_obj.localAllFishDetectionsFile, index_col=0)
    
    t_output = [poly.contains(Point(x*1296, y*972)) for x,y in zip(t_dt.xc, t_dt.yc)]
    t_dt['InBounds'] = t_output

    if t_dt.shape[0] == 0:
        print(projectID)
        continue

    num_movies = len(fm_obj.lp.movies)
    try:
        total_time = sum([x.endTime - x.startTime for x in fm_obj.lp.movies],datetime.timedelta(0))
    except:
        pdb.set_trace()
    total_detections = d_dt.shape[0]
    tracked_detections = t_dt.shape[0]
    
    track_lengths = t_dt.groupby('track_id').count()['base_name'].rename('track_length')
    t_dt = pd.merge(t_dt, track_lengths, left_on = 'track_id', right_on = track_lengths.index)
    t_dt['binned_track_length'] = t_dt.track_length.apply(bin_tracklength)
    b_tl = t_dt.groupby('binned_track_length').count()['track_length']
    b_tl = b_tl/b_tl.sum()

    track_lengths_by_project = track_lengths_by_project.append(b_tl.rename(projectID))
    try:
        temp = t_dt[t_dt.p_value > .7].groupby(['track_id', 'track_length', 'base_name']).mean()[['class_id', 'p_value','InBounds']].rename({'class_id':'SexCall'}, axis = 1).reset_index().sort_values(['base_name','track_id'])
        temp['projectID'] = projectID
        temp.to_csv(fm_obj.localAllTracksSummaryFile, index = False)
        fm_obj.uploadData(fm_obj.localAllTracksSummaryFile)
        sex_calls = sex_calls.append(temp)
    except:
        pdb.set_trace()

import seaborn as sns
import matplotlib.pyplot as plt

track_lengths_by_project.to_csv(fm_obj.localAnalysisStatesDir + 'TrackLengthDistribution.csv')
fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'TrackLengthDistribution.csv')
sex_calls.groupby(['projectID','BinnedSex']).count()['base_name'].reset_index().pivot_table(columns = 'projectID', index = 'BinnedSex', values = 'base_name').to_csv(fm_obj.localAnalysisStatesDir + 'SexCallDistribution.csv')
fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'SexCallDistribution.csv')
pdb.set_trace()

#rerun 'MC_singlenuc63_1_Tk9_060220'