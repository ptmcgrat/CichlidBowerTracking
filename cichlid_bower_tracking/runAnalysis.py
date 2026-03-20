import argparse, datetime, pdb, multiprocessing, random, subprocess, os
from helper_modules.file_manager import FileManager as FM
try:
	from itertools import batched
except ImportError:
	from more_itertools import batched

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
ma.add_argument('--NFilter', type=int, help='Optional argment filter out clips with less sand manipulation (200 might be a good threshold)', default = 0)

cfas = subparser.add_parser('DLCVideos', description = 'Create video clips to upload into DLC for annotation.')
cfas.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

train = subparser.add_parser('TrainModel', description = 'Train a 3D Resnet to automatically classify sand manipulation events using annotated data')
train.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
train.add_argument('--Exclude', type=str, nargs='+', help='The Analysis IDs to exclude', default = 100)

cc = subparser.add_parser('ClassifyClusters', description='Use created ML model to classify clusters for each project')
cc.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
cc.add_argument('--ModelID', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

trackfish = subparser.add_parser('TrackFish', description = 'Run YOLO to track and sex fish')
trackfish.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
trackfish.add_argument('--BatchID', type=int, help='Restrict the analysis to a subset that share the batchID')
trackfish.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

posefish = subparser.add_parser('PoseFish', description = 'Run YOLO to track and sex fish')
posefish.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
posefish.add_argument('--BatchID', type=int, help='Restrict the analysis to a subset that share the batchID')
posefish.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')

atwcfish = subparser.add_parser('AssociateTracksWithClusters', description = 'Identify fish responsible for manipulations')
atwcfish.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

summary = subparser.add_parser('Summary', description = 'Summarize all of the analyzed data')
summary.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

edit = subparser.add_parser('EditVideos', description = 'Add cluster boxes to videos that you annotated')
edit.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')

fa = subparser.add_parser('FixAnnotations', description = 'Manually fix sand manipulation categories. Creating this to fix the reflection/nofishother class.')
fa.add_argument('AnalysisID', type=str, help='Optional name of analysisID to restrict the analysis to.')
fa.add_argument('Category', type=str, help='The category you would like to fix')
fa.add_argument('Initials', type=str, help='Initials of person annotating the videos')

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
except AttributeError or KeyError:
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
	s_dt.to_csv(fm_obj.localSummaryFile, index = True)
	fm_obj.uploadData(fm_obj.localSummaryFile)

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
		#dp_obj.downloadProjectData()
		dp_obj.validateInputData()
		#dp_obj.createSmoothedArray()
		dp_obj.createDepthFigures()
		#dp_obj.createRGBVideo()
		dp_obj.uploadProjectData(delete = True)
		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'Depth'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

	"""
	writer = pypdf.PdfWriter()
	for projectID in s_dt[(s_dt.Depth == True)].index.to_list():
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
	"""
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

		videoIndices = [] if row.videoIDsToRun != row.videoIDsToRun or row.videoIDsToRun == 'VideoIndices: ' else row.videoIDsToRun.split(': ')[1].split(',')

		already_run = [] if row.Cluster == 'VideoIndices: ' else row.Cluster.split(': ')[1].split(',')
		
		videoIndices = [int(x) for x in videoIndices if x not in already_run]
		
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		for videoIndex in videoIndices:
			
			cp_obj = CP(fm_obj, videoIndex, workers)
			cp_obj.downloadProjectData()
			cp_obj.validateInputData()
			cp_obj.runClusterAnalysis()
			cp_obj.addCropAndDepthCoordinates()
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
		if row.videoIDsToAnnotate == 'VideoIndices: ' or row.videoIDsToAnnotate != row.videoIDsToAnnotate:
			print('Warning: No videos specified for this project. Skipping')
			continue
		else:
			videoIndices = [int(x) for x in row.videoIDsToAnnotate.split(': ')[1].split(',')]
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		fm_obj.setProjectID(projectID)
		mlv_obj = MLVP(fm_obj, args.Number, videoIndices, 'Videos')
		mlv_obj.downloadProjectData()
		mlv_obj.validateInputData()
		labeled_videos = mlv_obj.labelVideos(args.Initials, args.NFilter)
		quit = mlv_obj.uploadProjectData(delete = True)
		
		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
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
		if len(videoIndices) > 3:
			print('Warning: Cannot run more than 3 videos. Randomly picking 3')
			videoIndices = random.sample(videoIndices, 3)

		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		fm_obj.setProjectID(projectID)
		mlv_obj = MLVP(fm_obj, None, videoIndices, 'DLC')
		mlv_obj.downloadProjectData()
		mlv_obj.validateInputData()
		mlv_obj.createDLCVideos()
		mlv_obj.uploadProjectData(delete = True)

		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'DLCVideos'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)


elif args.AnalysisType == 'TrainModel':
	from data_preparers.threeD_model_preparer import ThreeDModelPreparer as TDMP
	tdm_obj = TDMP(fm_obj, args.Exclude)
	tdm_obj.downloadProjectData()
	tdm_obj.validateInputData()
	tdm_obj.create3DModel()
	tdm_obj.uploadData(delete = False)
	

elif args.AnalysisType == 'ClassifyClusters':
	from data_preparers.threeD_classifier_preparer import ThreeDClassifierPreparer as TDCP
	
	for projectID, row in s_dt.loc[projectIDs].iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)

		videoIndices = [] if row.videoIDsToRun != row.videoIDsToRun or row.videoIDsToRun == 'VideoIndices: ' else row.videoIDsToRun.split(': ')[1].split(',')
		videoIndices = [int(x) for x in videoIndices]
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)

		tdcp_obj = TDCP(fm_obj, videoIndices)
		tdcp_obj.downloadData()
		tdcp_obj.validateInputData()
		tdcp_obj.predictLabels()
		tdcp_obj.createSummaryFile()
		tdcp_obj.uploadData()

		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'ClassifyClusters'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

elif args.AnalysisType == 'TrackFish' or args.AnalysisType == 'PoseFish':
	from data_preparers.track_fish_preparer import TrackFishPreparer as TFP
	for projectID, row in s_dt.loc[projectIDs].iterrows():
		if projectID not in projectIDs:
			continue
		if args.BatchID is not None:
			if row.BatchID != args.BatchID:
				continue

		videoIndices = row.videoIDsToRun.split(': ')[1].split(',')
		already_run = [] if row[args.AnalysisType] == 'VideoIndices: ' else row[args.AnalysisType].split(': ')[1].split(',')
		videoIndices = [int(x) for x in videoIndices if x not in already_run]
		if len(videoIndices) == 0:
			continue
		fm_obj.setProjectID(projectID)

		print('Running: ' + projectID)
		print(' Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)	
		


		for sub_list in batched(videoIndices, 20):
			print('   Running: ' + ','.join([str(x) for x in sub_list]), flush = True)	
			
			tfp_obj = TFP(fm_obj, sub_list, args.AnalysisType)

			print('   Downloading data: ' + str(datetime.datetime.now()))
			tfp_obj.downloadProjectData()
			tfp_obj.validateInputData()
			print('   Tracking: ' + str(datetime.datetime.now()))
			tfp_obj.runYOLOAnalysis()
			print('   Uploading: ' + str(datetime.datetime.now()))		
			tfp_obj.uploadProjectData(delete = True)
			print('   Done: ' + str(datetime.datetime.now()))

		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt

		if s_dt.loc[projectID,args.AnalysisType] == 'VideoIndices: ':
			s_dt.loc[projectID,args.AnalysisType] +=  ','.join([str(x) for x in videoIndices])
		else:
			s_dt.loc[projectID,args.AnalysisType] +=  ',' + ','.join([str(x) for x in videoIndices])

		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

elif args.AnalysisType == 'AssociateTracksWithClusters':
	from data_preparers.associate_tracks_preparer import AssociateTracksPreparer as ATP
	for projectID, row in s_dt.loc[projectIDs].iterrows():
		
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
		fm_obj.setProjectID(projectID)
		videoIndices = row.TrackFish.split(': ')[1].split(',')
		videoIndices = [int(x) for x in videoIndices]

		atp_obj = ATP(fm_obj, videoIndices)
		atp_obj.downloadData()
		atp_obj.validateInputData()
		atp_obj.createAssociations()
		atp_obj.uploadData(delete=False)

		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'AssociateTracksWithClusters'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)


elif args.AnalysisType == 'Summary':
	from data_preparers.summary_preparer import SummaryPreparer as SP
	for projectID, row in s_dt.loc[projectIDs].iterrows():
		if projectID not in projectIDs:
			continue
		print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)

		fm_obj.setProjectID(projectID)
		sp_obj = SP(fm_obj)
		sp_obj.downloadData()
		sp_obj.validateInputData()
		sp_obj.createSummaryFigures()
		sp_obj.uploadData(delete=True)
		
		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'Summary'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

	fm_obj.createDirectory(fm_obj.localAnalysisFinalDataDir)
	for projectID, row in s_dt[s_dt.Summary == True].iterrows():
		fm_obj.setProjectID(projectID)
		for s_data in [fm_obj.localSummarizedClustersEvents,fm_obj.localSummarizedBuildingFigure,fm_obj.localSummarizedHourlyClusterFigure,fm_obj.localSummarizedHistogramFigure]:
			fm_obj.downloadData(s_data)
			subprocess.run(['mv', s_data, fm_obj.localAnalysisFinalDataDir + projectID + '__' + os.path.basename(s_data)])
	fm_obj.uploadData(fm_obj.localAnalysisFinalDataDir)

elif args.AnalysisType == 'EditVideos':
	from data_preparers.edit_videos_preparer import EditVideosPreparer as EVP
	for projectID, row in fm_obj.s_dt.iterrows():
		if projectID not in projectIDs:
			continue
		print(projectID)
		videoIndices2 = row.TrackFish.split(': ')[1].split(',')

		if row.videoIDsToAnnotate == 'VideoIndices: ' or row.videoIDsToAnnotate != row.videoIDsToAnnotate:
			print('Warning: No videos specified for this project. Skipping')
			continue
		else:
			videoIndices = [int(x) for x in row.videoIDsToAnnotate.split(': ')[1].split(',') if x in videoIndices2]
		#videoIndices = [0,1,2]
		print('Running: ' + ','.join([str(x) for x in videoIndices]), flush = True)
		fm_obj.setProjectID(projectID)
		evp_obj = EVP(fm_obj, videoIndices)
		evp_obj.downloadProjectData()
		evp_obj.validateInputData()
		evp_obj.editVideos()
		evp_obj.uploadProjectData(delete = False)
		
		fm_obj = FM(analysisID, projectID)
		s_dt = fm_obj.s_dt
		s_dt.loc[projectID,'EditVideos'] = True
		s_dt.to_csv(fm_obj.localSummaryFile, index = True)
		fm_obj.uploadData(fm_obj.localSummaryFile)

elif args.AnalysisType == 'FixAnnotations':
	from data_preparers.manual_label_video_fixer import ManualLabelVideoFixer as MLVF
	mlvf_obj = MLVF(fm_obj, args.Category, args.AnalysisID, projectIDs)
	mlvf_obj.downloadData()
	mlvf_obj.validateInputData()
	mlvf_obj.fixVideos(args.Initials)
	mlvf_obj.uploadData(delete = True)
#s_dt.to_csv(fm_obj.localSummaryFile, index = True)
#fm_obj.uploadData(fm_obj.localSummaryFile)
