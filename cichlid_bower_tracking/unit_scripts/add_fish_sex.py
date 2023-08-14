import argparse, subprocess
import sys
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 
from cichlid_bower_tracking.data_preparers.project_preparer import ProjectPreparer as PP

parser = argparse.ArgumentParser(usage = 'This script will use Pytorch Fish Sex Detection model')
parser.add_argument('ProjectID', type = str, help = 'Manually identify the project you want to analyze')
parser.add_argument('AnalysisID', type = str, help = 'Manually identify the project you want to analyze')
parser.add_argument('--VideoIndex', type = int, help = 'Index of video that should be analyzed')

args = parser.parse_args()

pp_obj = PP(projectID = args.ProjectID, analysisID = args.AnalysisID)
pp_obj.runMLFishDetection(args.VideoIndex)

