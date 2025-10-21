import argparse, datetime, pdb, multiprocessing
from helper_modules.file_manager import FileManager as FM

# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to analyze bower building data taken using PiCameras and Realsense Depth Sensors') 
subparser = parser.add_subparsers(required = True, title='Analysis Commands', dest='AnalysisType',
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

ma = subparser.add_parser('AnnotateVideos', description = 'Manually annotate sand manipulation videos into 10 categories. Restricts videos to VideoIDsToAnnotate column.')
ma.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
ma.add_argument('Initials', type=str, help='Initials of person annotating the videos')
ma.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to.')
ma.add_argument('--Number', type=int, help='Optional argment to specify how many videos per project to annotate', default = 100)

cfas = subparser.add_parser('CreateFrameAnnotationSet', description = 'Create folder of frames to upload into CVAT. Restricts videos to VideoIDsToAnnotate column.')
cfas.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
cfas.add_argument('--Number', type=int, help='Optional argment to specify how many videos per project to annotate', default = 100)

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

try:
	number = args.Number
except AttributeError:
	number = 0
try:
	projectIDs = fm_obj.getProjectIDs(args.AnalysisType, args.ProjectIDs, number)
except AttributeError:
	projectIDs = fm_obj.getProjectIDs(args.AnalysisType, None, number)

if 'RunAnalysis' not in fm_obj.s_dt:
	s_dt['RunAnalysis'] = True

if args.AnalysisType == 'AnalyzeStates':
	for projectID in projectIDs:
		print('Determining state for: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)
		out_data = fm_obj.getProjectStates()
		for k, v in out_data.items():
			if k not in s_dt:
				s_dt[k] = False
			s_dt.loc[projectID, k] = v

elif args.AnalysisType == 'Prep':
	import PyPDF2 as pypdf
	from data_preparers.prep_preparer import PrepPreparer as PrP
	
	print('The following projectIDs will be analyzed for ' + args.AnalysisType + ': ' + ','.join(projectIDs))
	
	print('Downloading all data')
	for projectID in projectIDs:
		print('Downloading data for: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID, print_issues = True)
		prp_obj = PrP(fm_obj)
		prp_obj.downloadProjectData()
		prp_obj.validateInputData()

	for projectID in projectIDs:
		print('Running prep for: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
		fm_obj.setProjectID(projectID)
		prp_obj.prepData()
		prp_obj.uploadProjectData(delete = False)
		s_dt.loc[projectID,'Prep'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

	writer = pypdf.PdfWriter()
	for projectID in s_dt[(s_dt.Prep == True)].index.sort_values().to_list():
		fm_obj.setProjectID(projectID)
		fm_obj.downloadData(fm_obj.trialOverviewFile)
		f = open(fm_obj.trialOverviewFile, 'rb')
		reader = pypdf.PdfReader(f)
		for page_number in range(len(reader.pages)):
			writer.add_page(reader.pages[page_number])
	with open(fm_obj.localAnalysisStatesDir + 'Collated_TrialPrepSummary.pdf', 'wb') as f:
		writer.write(f)
	fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'Collated_TrialPrepSummary.pdf')

elif args.AnalysisType == 'Depth':
	import PyPDF2 as pypdf
	from data_preparers.depth_preparer import DepthPreparer as DP

	print('The following projectIDs will be analyzed for ' + args.AnalysisType + ': ' + ','.join(projectIDs))

	for projectID in projectIDs:
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)
		dp_obj = DP(fm_obj)
		dp_obj.downloadProjectData()
		dp_obj.validateInputData()
		dp_obj.createSmoothedArray()
		dp_obj.createDepthFigures()
		dp_obj.createRGBVideo()
		dp_obj.uploadProjectData(delete = True)
		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'Depth'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

	writer = pypdf.PdfWriter()
	for projectID in s_dt[(s_dt.Depth == True)].index.sort_values().to_list():
		fm_obj.setProjectID(projectID)
		fm_obj.downloadData(fm_obj.localDailyDepthSummaryFigure)
		f = open(fm_obj.localDailyDepthSummaryFigure, 'rb')
		reader = pypdf.PdfReader(f)
		for page_number in range(len(reader.pages)):
			writer.add_page(reader.pages[page_number])
	with open(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf', 'wb') as f:
		writer.write(f)
	print('Finished analysis: ' + str(datetime.datetime.now()), flush = True)
	print(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')
	fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')

elif args.AnalysisType == 'Cluster':
	from data_preparers.cluster_preparer import ClusterPreparer as CP
	print('The following projectIDs will be analyzed for ' + args.AnalysisType + ': ' + ','.join(projectIDs))

	if args.Workers is None:
		workers = multiprocessing.cpu_count()
	else:
		workers = args.Workers

	for projectID, row in s_dt.loc[projectIDs].iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)

		if row.videoIDsToRun == 'VideoIndices: ':
			videoIndices = [] if row.videoIDsToRun == 'VideoIndices: ' else row.videoIDsToRun.split(': ')[1].split(',')
		else:
			videoIndices = row.videoIDsToRun.split(': ')[1].split(',')

		already_run = [] if row.Cluster == 'VideoIndices: ' else row.Cluster.split(': ')[1].split(',')
		videoIndices = [int(x) for x in videoIndices if x not in already_run]
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		for videoIndex in videoIndices:
			
			cp_obj = CP(fm_obj, videoIndex, workers)
			cp_obj.downloadProjectData()
			cp_obj.validateInputData()
			cp_obj.runClusterAnalysis()
			cp_obj.uploadProjectData(delete = True)

			fm_obj = FM(analysisID, projectID)
			s_dt = fm_obj.s_dt

			if s_dt.loc[projectID,'Cluster'] == 'VideoIndices: ':
				s_dt.loc[projectID,'Cluster'] +=  str(videoIndex)
			else:
				s_dt.loc[projectID,'Cluster'] +=  ',' + str(videoIndex)

			s_dt.to_csv(fm_obj.localSummaryFile, index = True)
			fm_obj.uploadData(fm_obj.localSummaryFile)


elif args.AnalysisType == 'AnnotateVideos':
	from data_preparers.manual_label_video_preparer import ManualLabelVideoPreparer as MLVP
	print('The following projectIDs will be analyzed for ' + args.AnalysisType + ': ' + ','.join(projectIDs))

	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print(projectID)
		if row.videoIDsToAnnotate == 'VideoIndices: ':
			print('Warning: No videos specified for this project. Skipping')
			continue
		else:
			videoIndices = [int(x) for x in row.videoIDsToAnnotate.split(': ')[1].split(',')]
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		fm_obj.setProjectID(projectID)
		mlv_obj = MLVP(fm_obj, args.Number, videoIndices, 'Videos')
		mlv_obj.downloadProjectData()
		mlv_obj.validateInputData()
		labeled_videos = mlv_obj.labelVideos(args.Initials)
		quit = mlv_obj.uploadProjectData(delete = True)
		
		s_dt.loc[projectID,'ManualAnnotation'] = labeled_videos
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

		if quit:
			break

elif args.AnalysisType == 'DLCVideos':
	from data_preparers.manual_label_video_preparer import ManualLabelVideoPreparer as MLVP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print(projectID)
		if row.videoIDsToAnnotate == 'VideoIndices: ':
			print('Warning: No videos specified for this project. Skipping')
			continue
		else:
			videoIndices = [int(x) for x in row.videoIDsToAnnotate.split(': ')[1].split(',')]
		if len(videoIndices) != 3:
			print('Warning: Need to specify three and exactly three videos. Skipping')
			continue
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		fm_obj.setProjectID(projectID)
		mlv_obj = MLVP(fm_obj, args.Number, videoIndices, 'DLC')
		mlv_obj.downloadProjectData()
		mlv_obj.validateInputData()
		mlv_obj.createDLCVideos()
		mlv_obj.uploadProjectData(delete = True)

		s_dt.loc[projectID,'DLCVideos'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)


elif args.AnalysisType == 'TrainModel':
	from cichlid_bower_tracking.data_preparers.threeD_model_preparer import ThreeDModelPreparer as TDMP
	if (s_dt.AnnotateVideos == False).sum() != 0:
		print('Warning: You are training a model even though all projects have not been annotated')
	tdm_obj = TDMP(fm_obj, modelID)
	tdm_obj.validateInputData()
	tdm_obj.create3DModel()

elif args.AnalysisType == 'ClassifyClusters':
	from cichlid_bower_tracking.data_preparers.threeD_classifier_preparer import ThreeDClassifierPreparer as TDCP
	projectIDs = args.ProjectIDs if args.ProjectIDs is not None else s_dt[(s_dt.RunAnalysis == True) & (s_dt.Cluster == True) & (s_dt[args.AnalysisType] == False)].index.to_list()
	print('The following projectIDs will be analyzed for ' + args.AnalysisType + ': ' + ','.join(projectIDs))

	for projectID, row in fm_obj.s_dt.iterrows():

		tdcp_obj = TDCP(self.fileManager, modelID)
		tdcp_obj.validateInputData()
		tdcp_obj.predictLabels()
		tdcp_obj.createSummaryFile()

s_dt.to_csv(fm_obj.localSummaryFile, index = True)
fm_obj.uploadData(fm_obj.localSummaryFile)
