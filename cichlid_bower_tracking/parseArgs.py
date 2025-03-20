import argparse, datetime, pdb, git

from helper_modules.file_manager import FileManager as FM

branch_name = git.Repo().head.ref.name
# Create arguments for the script
#parent_parser = argparse.ArgumentParser(add_help = False, description='This script is used to analyze bower building data taken using PiCameras and Realsense Depth Sensors')
#parent_parser.add_argument('AnalysisID', type = str, help = 'AnalysisID you would like to analyze')

parser = argparse.ArgumentParser(description='This script is used to analyze bower building data taken using PiCameras and Realsense Depth Sensors') 
subparser = parser.add_subparsers(required = True, title='Analysis Commands',
                                   description='These are the valid commands that you can run')

a_st = subparser.add_parser('AnalyzeStates', description = 'This is used to update the AnalysisStates.csv with the current state of analysis for each projectID')
a_st.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

prep = subparser.add_parser('Prep', description = 'This is used to crop and register the depth and video data')
prep.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
prep.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

depth = subparser.add_parser('Depth', description = 'Analyze all of the depth data')
depth.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
depth.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

cluster = subparser.add_parser('Cluster', description = 'Run HMM and DBScan Cluster analysis to identify sand manipulation events')
cluster.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
cluster.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
cluster.add_argument('--Workers', type=int, help='Number of workers')

ma = subparser.add_parser('ManualAnnotation', description = 'Manually annotate sand manipulation videos into 10 categories')
ma.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
ma.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
ma.add_argument('--Number', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
ma.add_argument('--Initials', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

train = subparser.add_parser('TrainModel', description = 'Train a 3D Resnet to automatically classify sand manipulation events using annotated data')
train.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
train.add_argument('--ModelID', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

cc = subparser.add_parser('ClassifyClusters', description='Use created ML model to classify clusters for each project')
cc.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
cc.add_argument('--ModelID', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

summary = subparser.add_parser('Summary', description = 'Summarize all of the analyzed data')
summary.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

args = parser.parse_args()

analysisID = args.AnalysisID
pdb.set_trace()
fm_obj = FM(analysisID)
