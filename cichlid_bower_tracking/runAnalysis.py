import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path
from helper_modules.file_manager import FileManager as FM
import math

def is_nan(value):
    if isinstance(value, float):
        return math.isnan(value)
    elif isinstance(value, str):
        return value.lower() == "nan"
    return False

# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type=str, choices=['Check', 'Prep', 'Depth', 'Cluster', 'ClusterClassification', 'TrackFish', 'AssociateClustersWithTracks', 'Summary'], help='Type of analysis to run')
parser.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
parser.add_argument('--Workers', type=int, help='Number of workers')
args = parser.parse_args()

analysisID = 'MC_multi'
# Identify projects to run analysis on
fm_obj = FM()
fm_obj.downloadData(fm_obj.localSummaryFile)
# pdb.set_trace()
if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
	print('Cant find ' + fm_obj.localSummaryFile)
	sys.exit()

if args.AnalysisType != 'Cluster':
    args.AnalysisType = 'Cluster'

fm_obj.s_dt.columns = [c.replace(' ', '_') for c in fm_obj.s_dt.columns]
fm_obj.s_dt['StartTime'] = None
fm_obj.s_dt['EndTime'] = None
num_workers = args.Workers
# fm_obj.s_dt.reset_index(inplace=True)

# for subjectID, row in fm_obj.s_dt.iterrows():
# 	# pdb.set_trace()
# 	for projectID in row.ProjectIDs.split(',,'):

# 		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

# 		fm_obj.setProjectID(subjectID, projectID)
# 		if args.AnalysisType == 'Prep':
# 			from data_preparers.prep_preparer import PrepPreparer as PrP
# 			prp_obj = PrP(fm_obj)
# 			prp_obj.downloadProjectData()
# 			prp_obj.validateInputData()
# 			prp_obj.prepData()
# 			prp_obj.uploadProjectData(delete = False)

# 		elif args.AnalysisType == 'Depth':
# 			from data_preparers.depth_preparer import DepthPreparer as DP
# 			dp_obj = DP(fm_obj)
# 			#dp_obj.downloadProjectData()
# 			dp_obj.validateInputData()
# 			dp_obj.createSmoothedArray()
# 			dp_obj.createDepthFigures()
# 				#dp_obj.createRGBVideo()
# 			dp_obj.uploadProjectData(delete = False)

# 		elif args.AnalysisType == 'Cluster':
# 			from data_preparers.cluster_preparer import ClusterPreparer as CP
# 			for videoIndex in row.VideoIDs.split(': ')[1].split(','):
# 				FMObj = FM(projectID=projectID)
# 				cp_obj = CP(FMObj, videoIndex)
# 				cp_obj.downloadProjectData()
# 				cp_obj.validateInputData()
# 				cp_obj.runClusterAnalysis()
# 				cp_obj.uploadProjectData(delete = False)

for projectID, row in fm_obj.s_dt.iterrows():

		if(is_nan(row.VideoIDs_new)):
			continue

		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
		pdb.set_trace()
		fm_obj.setProjectID(row.SubjectID, projectID)

		if args.AnalysisType == 'Prep':
			from data_preparers.prep_preparer import PrepPreparer as PrP
			prp_obj = PrP(fm_obj)
			prp_obj.downloadProjectData()
			prp_obj.validateInputData()
			prp_obj.prepData()
			prp_obj.uploadProjectData(delete = False)

		elif args.AnalysisType == 'Depth':
			from data_preparers.depth_preparer import DepthPreparer as DP
			dp_obj = DP(fm_obj)
			#dp_obj.downloadProjectData()
			dp_obj.validateInputData()
			dp_obj.createSmoothedArray()
			dp_obj.createDepthFigures()
				#dp_obj.createRGBVideo()
			dp_obj.uploadProjectData(delete = False)

		elif args.AnalysisType == 'Cluster':
			from data_preparers.cluster_preparer import ClusterPreparer as CP
			for videoIndex in row.VideoIDs_new.split(': ')[1].split(','):
				cp_obj = CP(fm_obj, int(videoIndex),num_workers)
				pdb.set_trace()
				cp_obj.downloadProjectData()
				pdb.set_trace()
				cp_obj.validateInputData()
				pdb.set_trace()
				cp_obj.runClusterAnalysis()
				pdb.set_trace()
				cp_obj.uploadProjectData(delete = False)
				pdb.set_trace()

if args.AnalysisType == 'Depth':
	import PyPDF2 as pypdf
	writer = pypdf.PdfFileWriter()
	for subjectID, row in fm_obj.s_dt.iterrows():
		for projectID in row.ProjectIDs.split(',,'):
			fm_obj.setProjectID(subjectID, projectID)
			f = open(fm_obj.localDailyDepthSummaryFigure, 'rb')
			reader = pypdf.PdfFileReader(f)
			for page_number in range(reader.numPages):
				writer.addPage(reader.getPage(page_number))
	with open(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf', 'wb') as f:
		writer.write(f)
	print('Finished analysis: ' + str(datetime.datetime.now()), flush = True)
	print(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')
	fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')

