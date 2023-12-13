import argparse, subprocess, pdb, datetime, os, sys,multiprocessing
import pandas as pd
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path
from helper_modules.file_manager import FileManager as FM


# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type=str, choices=['Check', 'Prep', 'Depth', 'Cluster', 'ClusterClassification', 'TrackFish', 'AssociateClustersWithTracks', 'Summary'], help='Type of analysis to run')
parser.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
parser.add_argument('--Workers', type=int, help='Number of workers')
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM()
fm_obj.downloadData(fm_obj.localSummaryFile)

if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
	print('Cant find ' + fm_obj.localSummaryFile)
	sys.exit()

# Filter to projects that should be run
projects = fm_obj.s_dt[(fm_obj.s_dt.RunAnalysis.str.upper() == 'TRUE') & (fm_obj.s_dt[args.AnalysisType].str.upper() == 'FALSE')]
for projectID,row in projects.iterrows():

	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

	fm_obj.setProjectID(projectID)
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
		#dp_obj.createSmoothedArray()
		dp_obj.createDepthFigures()
			#dp_obj.createRGBVideo()
		#dp_obj.uploadProjectData(delete = False)

	elif args.AnalysisType == 'Cluster':
		from data_preparers.cluster_preparer import ClusterPreparer as CP
		cp_obj = CP(fm_obj,multiprocessing.cpu_count()
)
		#cp_obj.downloadProjectData()
		cp_obj.validateInputData()
		videos = list(range(len(fm_obj.lp.movies)))
		for videoIndex in videos:
			cp_obj.runClusterAnalysis(videoIndex)


if args.AnalysisType == 'Depth':
	import PyPDF2 as pypdf
	writer = pypdf.PdfFileWriter()
	for projectID,row in projects.iterrows():
		fm_obj.setProjectID(projectID)
		f = open(fm_obj.localDailyDepthSummaryFigure, 'rb')
		reader = pypdf.PdfFileReader(f)
		for page_number in range(reader.numPages):
			writer.addPage(reader.getPage(page_number))
	with open(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf', 'wb') as f:
		writer.write(f)
	print('Finished analysis: ' + str(datetime.datetime.now()), flush = True)
	print(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')
	fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')

