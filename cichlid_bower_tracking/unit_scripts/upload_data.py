# Things to add

import argparse,sys
from cichlid_bower_tracking.data_preparers.project_preparer import ProjectPreparer as PP

parser = argparse.ArgumentParser()
parser.add_argument('DataType', type = str, choices=['Prep','Depth','Cluster','ClusterClassification', 'TrackFish', 'Train3DResnet', 'ManualLabelVideos','Summary','All', 'AddFishSex', 'AssociateClustersWithTracks', 'AssociateClustersWithSex'], help = 'What type of analysis to perform')
parser.add_argument('--ProjectID', type = str, help = 'Manually identify the projects you want to analyze. If All is specified, all non-prepped projects will be analyzed')
parser.add_argument('--AnalysisID', type = str, help = 'The ID of the analysis state this project belongs to')
parser.add_argument('--VideoIndex', type = int, help = 'Specify which video should be downloaded if Cluster analysis is to be performed')
parser.add_argument('--ModelID', type = str, help = 'Specify a ModelID to download')
parser.add_argument('--Delete', action = 'store_true', help = 'Delete data once it is uploaded')
parser.add_argument('--NoUpload', action = 'store_true', help = 'Do not upload data (but still delete it)')

args = parser.parse_args()

pp_obj = PP(args.ProjectID, args.ModelID, analysisID = args.AnalysisID)
pp_obj.uploadData(args.DataType, args.VideoIndex, args.Delete, args.NoUpload)

