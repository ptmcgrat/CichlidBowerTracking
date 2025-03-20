import argparse, datetime, pdb
import PyPDF2 as pypdf
from helper_modules.file_manager import FileManager as FM

# Create arguments for the script
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



# Identify projects to run analysis on
fm_obj = FM(analysisID)
s_dt = fm_obj.s_dt

projectIDs = args.ProjectIDs if args.ProjectIDs is not None else fm_obj.s_dt['ProjectIDs']
pdb.set_trace()

if args.AnalysisType == 'AnalyzeStates':
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
		if 'RunAnalysis' not in fm_obj.s_dt:
			fm_obj.s_dt[k] = True

		fm_obj.setProjectID(projectID)
		out_data = fm_obj.getProjectStates()

		for k, v in out_data.items():
			if k not in fm_obj.s_dt:
				fm_obj.s_dt[k] = False
			fm_obj.s_dt.loc[dt.projectID == projectID, k] = v

	fm_obj.s_dt.to_csv(fm_obj.s_dt.localSummaryFile, index = True)
	fm_obj.uploadData(fm_obj.s_dt.localSummaryFile)

elif args.AnalysisType == 'Prep':
	from data_preparers.prep_preparer import PrepPreparer as PrP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)
		prp_obj = PrP(fm_obj)
		prp_obj.downloadProjectData()
		prp_obj.validateInputData()
		prp_obj.prepData()
		prp_obj.uploadProjectData(delete = False)

elif args.AnalysisType == 'Depth':
	from data_preparers.depth_preparer import DepthPreparer as DP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)
		dp_obj = DP(fm_obj)
		dp_obj.downloadProjectData()
		dp_obj.validateInputData()
		dp_obj.createSmoothedArray()
		dp_obj.createDepthFigures()
			#dp_obj.createRGBVideo()
		dp_obj.uploadProjectData(delete = False)

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


elif args.AnalysisType == 'Cluster':
	from data_preparers.cluster_preparer import ClusterPreparer as CP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)

		if ':' in row.VideoIDs:
			videoIndices = row.VideoIDs_new.split(': ')[1].split(',')
		else:
			videoIndices = range(range(len(fm_obj.lp.movies)))

		for videoIndex in videoIndices:
	
			cp_obj = CP(fm_obj, int(videoIndex),num_workers)
			cp_obj.downloadProjectData()
			cp_obj.validateInputData()
			cp_obj.runClusterAnalysis()
			cp_obj.uploadProjectData(delete = False)

elif args.AnalysisType == 'AnnotateVideos':
	from cichlid_bower_tracking.data_preparers.manual_label_video_preparer import ManualLabelVideoPreparer as MLVP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		fm_obj.setProjectID(projectID)
		mlv_obj = MLVP(fm_obj, args.Initials, args.Number)
		mlv_obj.validateInputData()
		mlv_obj.labelVideos()

elif args.AnalysisType == 'TrainModel':
	from cichlid_bower_tracking.data_preparers.threeD_model_preparer import ThreeDModelPreparer as TDMP
	for projectID, row in fm_obj.s_dt.iterrows():

		fm_obj.setProjectID(projectID)
		tdm_obj = TDMP(fm_obj, modelID)
		tdm_obj.validateInputData()
		tdm_obj.create3DModel()