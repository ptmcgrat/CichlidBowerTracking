import argparse, pdb, datetime
import pandas as pd

from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
#test

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

    pdb.set_trace()
