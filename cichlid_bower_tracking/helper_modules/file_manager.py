import os, subprocess, pdb, platform, shutil
from cichlid_bower_tracking.helper_modules.log_parser import LogParser as LP

class FileManager():
	def __init__(self, projectID = None, modelID = None, analysisID=None, rcloneRemote = 'CichlidPiData:', masterDir = 'BioSci-McGrath/Apps/CichlidPiData/'):
		# Identify directory for temporary local files
		if platform.node() == 'raspberrypi' or 'Pi' in platform.node() or 'bt-' in platform.node() or 'sv-' in platform.node():
			self._identifyPiDirectory()
		elif platform.node() == 'ebb-utaka.biosci.gatech.edu':
			self.localMasterDir = '/mnt/Storage/' + os.getenv('USER') + '/Temp/CichlidAnalyzer/'
		else:
			self.localMasterDir = os.getenv('HOME').rstrip('/') + '/' + 'Temp/CichlidAnalyzer/'

		# Identify cloud directory for rclone
		self.rcloneRemote = rcloneRemote
		# On some computers, the first directory is McGrath, on others it's BioSci-McGrath. Use rclone to figure out which
		output = subprocess.run(['rclone', 'lsf', self.rcloneRemote + masterDir], capture_output = True, encoding = 'utf-8')
		if output.stderr == '':
			self.cloudMasterDir = self.rcloneRemote + masterDir
		else:
			output = subprocess.run(['rclone', 'lsf', self.rcloneRemote + 'BioSci-' + masterDir], capture_output = True, encoding = 'utf-8')
			if output.stderr == '':
				self.cloudMasterDir = self.rcloneRemote + 'BioSci-' + masterDir
			else:
				self.cloudMasterDir = self.rcloneRemote + masterDir
				#raise Exception('Cant find master directory (' + masterDir + ') in rclone remote (' + rcloneRemote + '')

		self.analysisID = analysisID
		if analysisID is not None:
			self.localAnalysisStatesDir = self.localMasterDir + '__AnalysisStates/' + analysisID + '/'
			self.localSummaryFile = self.localAnalysisStatesDir + analysisID + '.csv'
		else:
			self.localEuthData = None
	

		if projectID is not None:
			self.createProjectData(projectID)

		self.modelID = modelID
		self.localMLDir = self.localMasterDir + '__MachineLearningModels/'
		
		self.createMLData(modelID)

		# Create file names and parameters
		self.createPiData()
		self.createAnnotationData()

		self._createParameters()

	def createPiData(self):
		self.localCredentialDir = self.localMasterDir + '__CredentialFiles/'
		self.localCredentialSpreadsheet = self.localCredentialDir + 'SAcredentials_1.json'
		self.localCredentialDrive = self.localCredentialDir +  'DriveCredentials.txt'
		self.localEmailCredentialFile = self.localCredentialDir + 'iof_credentials/sendgrid_key.secret'

	def getAllProjectIDs(self):
		output = subprocess.run(['rclone', 'lsf', '--dirs-only', self.cloudMasterDir], capture_output = True, encoding = 'utf-8')
		projectIDs = [x.rstrip('/') for x in output.stdout.split('\n') if x[0:2] != '__']
		return projectIDs

	def getProjectStates(self):

		# Dictionary to hold row of data
		row_data = {'projectID':self.projectID, 'tankID':'', 'StartingFiles':False, 'Prep':False, 'Depth':False, 'Cluster':False, 'ClusterClassification':False, 'TrackFish': False, 'Summary': False}

		# List the files needed for each analysis
		necessaryFiles = {}
		necessaryFiles['StartingFiles'] = [self.localLogfile, self.localPrepDir, self.localFrameTarredDir, self.localVideoDir, self.localFirstFrame, self.localLastFrame, self.localPiRGB, self.localFirstDepthRGB, self.localLastDepthRGB]
		necessaryFiles['Prep'] = [self.localDepthCropFile,self.localTransMFile,self.localVideoCropFile]
		necessaryFiles['Depth'] = [self.localSmoothDepthFile]
		necessaryFiles['Cluster'] = [self.localAllClipsDir, self.localManualLabelClipsDir, self.localManualLabelFramesDir]
		necessaryFiles['ClusterClassification'] = [self.localAllLabeledClustersFile]
		necessaryFiles['TrackFish'] = [self.localAllFishDetectionsFile, self.localAllFishTracksFile]

		necessaryFiles['Summary'] = [self.localSummaryDir]

		print('Checking project ' + self.projectID + ': ', end = '')
		# Try to download and read logfile
		try:
			self.downloadData(self.localLogfile)
		except FileNotFoundError:
			print('No Logfile. Continuing...')
			row_data['StartingFiles'] = False
			return row_data

		self.lp = LP(self.localLogfile)
		if self.lp.malformed_file:
			row_data['StartingFiles'] = False
			print('Malformed Log File. Continuing...')
			return row_data

		# Get additional files necessary for analysis based on videos
		for index,vid_obj in enumerate(self.lp.movies):
			vid_obj = self.returnVideoObject(index)
			necessaryFiles['StartingFiles'].append(vid_obj.localVideoFile)
			necessaryFiles['Cluster'].append(vid_obj.localLabeledClustersFile)
			necessaryFiles['Cluster'].append(vid_obj.localAllClipsDir[:-1] + '.tar')
			necessaryFiles['Cluster'].append(vid_obj.localManualLabelClipsDir[:-1] + '.tar')
			necessaryFiles['Cluster'].append(vid_obj.localManualLabelFramesDir[:-1] + '.tar')

		row_data['tankID'] = self.lp.tankID
		# Check if files exists

		directories = {}

		print(subprocess.run(['rclone','size',necessaryFiles['TrackFish'][0].replace(self.localMasterDir, self.cloudMasterDir)], capture_output = True, encoding = 'utf-8').stdout)
		print(subprocess.run(['rclone','size',necessaryFiles['TrackFish'][1].replace(self.localMasterDir, self.cloudMasterDir)], capture_output = True, encoding = 'utf-8').stdout)

		for analysis_type, analysis_files in necessaryFiles.items():
			for analysis_file in analysis_files:
				directories[os.path.dirname(os.path.realpath(analysis_file))] = []

		for local_path in directories:
			cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)
			output = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8')
			directories[local_path] = [x.rstrip('/') for x in output.stdout.split('\n')]

		for analysis_type, local_files in necessaryFiles.items():
			row_data[analysis_type] = all([os.path.basename(x) in directories[os.path.dirname(os.path.realpath(x))] for x in local_files])
		print('Individual file info added to csv file')
		return row_data
		


	def createProjectData(self, projectID):



		self.createAnnotationData()
		self.projectID = projectID
		if self.analysisID is None:
			self.localProjectDir = self.localMasterDir + '__ProjectData/' + projectID + '/'
		else:
			self.localProjectDir = self.localMasterDir + '__ProjectData/' + self.analysisID + '/' + projectID + '/'

		# Create logfile
		self.localLogfile = self.localProjectDir + 'Logfile.txt'
		self.localLogfileDir = self.localProjectDir + 'Logfiles/'
		self.localPrepLogfile = self.localLogfileDir + 'PrepLog.txt'
		self.localDepthLogfile = self.localLogfileDir + 'DepthLog.txt'
		self.localClusterClassificationLogfile = self.localLogfileDir + 'ClassifyLog.txt'
		
		# Data directories created by tracker
		self.localPrepDir = self.localProjectDir + 'PrepFiles/'
		self.localFrameDir = self.localProjectDir + 'Frames/'
		self.localFrameTarredDir = self.localProjectDir + 'Frames.tar'
		self.localVideoDir = self.localProjectDir + 'Videos/'
		self.localBackupDir = self.localProjectDir + 'Backups/'
		self.localFirstFrame = self.localPrepDir + 'FirstDepth.npy'
		self.localLastFrame = self.localPrepDir + 'LastDepth.npy'
		self.localPiRGB = self.localPrepDir + 'PiCameraRGB.jpg'
		self.localFirstDepthRGB = self.localPrepDir + 'FirstDepthRGB.jpg' 
		self.localLastDepthRGB = self.localPrepDir + 'LastDepthRGB.jpg'


		# Directories created by analysis
		self.localAnalysisDir = self.localProjectDir + 'MasterAnalysisFiles/'
		self.localSummaryDir = self.localProjectDir + 'Summary/'
		self.localAllClipsDir = self.localProjectDir + 'AllClips/'
		self.localManualLabelClipsDir = self.localProjectDir + 'MLClips/'
		self.localManualLabelFramesDir = self.localProjectDir + 'MLFrames/'
		self.localTroubleshootingDir = self.localProjectDir + 'Troubleshooting/'
		self.localTempDir = self.localProjectDir + 'Temp/'
		self.localPaceDir = self.localProjectDir + 'Pace/'

		# Files created by prep preparer
		self.localDepthCropFile = self.localAnalysisDir + 'DepthCrop.txt'
		self.localTransMFile = self.localAnalysisDir + 'TransMFile.npy'
		self.localVideoCropFile = self.localAnalysisDir + 'VideoCrop.txt'
		
		self.localPrepSummaryFigure = self.localSummaryDir + 'PrepSummary.pdf'
		self.localOldVideoCropFile = self.localAnalysisDir + 'VideoPoints.npy'

		# Files created by depth preparer
		self.localSmoothDepthFile = self.localAnalysisDir + 'smoothedDepthData.npy'
		self.localRGBDepthVideo = self.localAnalysisDir + 'DepthRGBVideo.mp4'
		self.localRawDepthFile = self.localTroubleshootingDir + 'rawDepthData.npy'
		self.localInterpDepthFile = self.localTroubleshootingDir + 'interpDepthData.npy'
		self.localDepthSummaryFile = self.localSummaryDir + 'DataSummary.xlsx'
		self.localDailyDepthSummaryFigure = self.localSummaryDir + 'DailyDepthSummary.pdf'
		self.localHourlyDepthSummaryFigure = self.localSummaryDir + 'HourlyDepthSummary.pdf'

		# Files created by cluster classifier preparer
		self.localTempClassifierDir = self.localProjectDir + 'TempClassifier/'
		self.localAllLabeledClustersFile = self.localAnalysisDir + 'AllLabeledClusters.csv'

		# Files created by fish_tracking preparer
		self.localAllFishTracksFile = self.localAnalysisDir + 'AllTrackedFish.csv'
		self.localAllFishDetectionsFile = self.localAnalysisDir + 'AllDetectionsFish.csv'
		self.localAllTracksSummaryFile = self.localAnalysisDir + 'AllSummarizedTracks.csv'

		# Files created by manual labelerer  preparers
		self.localNewLabeledFramesFile = self.localTempDir + 'NewLabeledFrames.csv'
		self.localNewLabeledFramesDir = self.localTempDir + 'NewLabeledFrames/'
		self.localNewLabeledVideosFile = self.localTempDir + 'NewLabeledVideos.csv'
		self.localNewLabeledClipsDir = self.localTempDir + 'NewLabeledClips/'


		self.localLabeledClipsProjectDir = self.localLabeledClipsDir + projectID + '/'
		self.localLabeledFramesProjectDir = self.localBoxedFishDir + projectID + '/'

		# Files created by summary preparer

		# miscellaneous files
		self.localPiErrorFile = self.localProjectDir + 'PiErrors.txt'

		try:
			self.downloadData(self.localLogfile)
			self.lp = LP(self.localLogfile)
		except FileNotFoundError:
			#print('No logfile created yet for ' + projectID)
			pass 

	def createMLData(self, modelID = None):
		if modelID is None and self.analysisID is None:
			return

		self.vModelID = modelID

		if modelID is None:
			self.vModelID = self.analysisID
			self.localYolov5WeightsFile = self.localMLDir + 'YOLOV5/' + self.analysisID + '/best.pt'

		else:
			self.localYolov5WeightsFile = self.localMLDir + 'YOLOV5/' + modelID + '/best.pt'

		self.local3DModelDir = self.localMLDir + 'VideoModels/' + self.vModelID + '/'
		self.local3DModelTempDir = self.localMLDir + 'VideoModels/' + self.vModelID + 'Temp/'

		self.localVideoModelFile = self.local3DModelDir + 'model.pth'
		self.localVideoClassesFile = self.local3DModelDir + 'classInd.txt'
		self.localModelCommandsFile = self.local3DModelDir + 'commands.log'
		self.localVideoProjectsFile = self.local3DModelDir + 'videoToProject.csv'
		self.localVideoLabels = self.local3DModelDir + 'confusionMatrix.csv'

	def createAnnotationData(self):
		self.localAnnotationDir = self.localMasterDir + '__AnnotatedData/'
		self.localObjectDetectionDir = self.localAnnotationDir + 'BoxedFish/'
		self.local3DVideosDir = self.localAnnotationDir + 'LabeledVideos/'

		self.localLabeledClipsFile = self.local3DVideosDir + 'ManualLabels.csv'
		self.localLabeledClipsDir = self.local3DVideosDir + 'Clips/'

		self.localBoxedFishFile = self.localObjectDetectionDir + 'BoxedFish.csv'
		self.localBoxedFishDir = self.localObjectDetectionDir + 'BoxedImages/'

	def downloadProjectData(self, dtype, videoIndex = None):

		if dtype == 'Prep':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localSummaryDir)
			self.createDirectory(self.localLogfileDir)
			self.downloadData(self.localPrepDir)
			self.downloadData(self.localLogfile)

		elif dtype == 'Depth':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localLogfileDir)
			self.createDirectory(self.localSummaryDir)

			#self.createDirectory(self.localPaceDir)

			self.downloadData(self.localLogfile)
			self.downloadData(self.localFrameDir, tarred = True)
			self.downloadData(self.localDepthCropFile)

		elif dtype == 'Cluster':
			#self.createMLData()
			self.createDirectory(self.localLogfileDir)
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localTempDir)
			self.createDirectory(self.localAllClipsDir)
			self.createDirectory(self.localManualLabelClipsDir)
			self.createDirectory(self.localManualLabelFramesDir)
			#self.createDirectory(self.localPaceDir)

			self.downloadData(self.localLogfile)
			if videoIndex is not None:
				videoObj = self.returnVideoObject(videoIndex)
				print('Downloading video ' + str(videoIndex))
				self.downloadData(videoObj.localVideoFile)
			else:
				print('Downloading video ' + self.localVideoDir)
				self.downloadData(self.localVideoDir)


		elif dtype == 'ClusterClassification':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localLogfileDir)

			self.downloadData(self.localLogfile)
			self.downloadData(self.localAllClipsDir, tarred_subdirs = True)
			self.downloadData(self.localAnalysisDir)
			self.downloadData(self.localTroubleshootingDir)
			if self.modelID is not None:
				self.downloadData(self.local3DModelDir)
			#self.createDirectory(self.localPaceDir)

		elif dtype == 'TrackFish':
			self.createDirectory(self.localLogfileDir)
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localTempDir)

			self.downloadData(self.localLogfile)
			self.downloadData(self.localYolov5WeightsFile)

			if videoIndex is not None:
				videoObj = self.returnVideoObject(videoIndex)
				print('Downloading video ' + str(videoIndex))
				self.downloadData(videoObj.localVideoFile)
			else:
				print('Downloading video ' + self.localVideoDir)
				self.downloadData(self.localVideoDir)

		elif dtype == 'AssociateClustersWithTracks':
			self.createDirectory(self.localLogfileDir)
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localAllFishDetectionsFile)
			self.downloadData(self.localAllFishTracksFile)
			self.downloadData(self.localOldVideoCropFile)
			self.downloadData(self.localAllLabeledClustersFile)

		elif dtype == 'Train3DResnet':
			self.createDirectory(self.local3DModelDir)
			self.createDirectory(self.local3DModelTempDir)
			self.createDirectory(self.localMasterDir)
			self.downloadData(self.localLabeledClipsFile)
			self.downloadData(self.localLabeledClipsDir, tarred_subdirs = True)		
			
		elif dtype == 'ManualLabelVideos':

			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localNewLabeledClipsDir)
			self.downloadData(self.localManualLabelClipsDir, tarred_subdirs = True)
			self.downloadData(self.localLabeledClipsFile)

		elif dtype == 'ManualLabelFrames':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localNewLabeledFramesDir)
			self.downloadData(self.localManualLabelFramesDir, tarred_subdirs = True)
			self.downloadData(self.localBoxedFishFile)

		elif dtype == 'Summary':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.localSummaryDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localAnalysisDir)
			self.downloadData(self.localPaceDir, allow_errors=True, quiet=True)
			self.downloadData(self.localEuthData, allow_errors=True, quiet=True)
			self.downloadData(self.localSummaryDir, allow_errors=True, quiet=True)

		elif dtype == 'All':
			self.createDirectory(self.localMasterDir)
			self.createDirectory(self.local3DModelDir)
			self.createDirectory(self.localAnalysisDir)
			self.createDirectory(self.localTroubleshootingDir)
			self.createDirectory(self.localTempDir)
			self.createDirectory(self.localAllClipsDir)
			self.createDirectory(self.localManualLabelClipsDir)
			self.createDirectory(self.localManualLabelFramesDir)
			self.createDirectory(self.localManualLabelFramesDir[:-1] + '_pngs')
			self.createDirectory(self.localPaceDir)
			self.downloadData(self.localLogfile)
			self.downloadData(self.localVideoDir)
			self.downloadData(self.localFrameDir, tarred = True)
			self.downloadData(self.local3DModelDir)
			self.downloadData(self.localEuthData, allow_errors=True, quiet=True)

		else:
			raise KeyError('Unknown key: ' + dtype)

	def uploadProjectData(self, dtype, videoIndex, delete, no_upload):
		if dtype == 'Prep':
			if not no_upload:
				self.uploadData(self.localDepthCropFile)
				self.uploadData(self.localTransMFile)
				self.uploadData(self.localVideoCropFile)
				self.uploadData(self.localPrepSummaryFigure)
				self.uploadData(self.localPrepLogfile)

			if delete:
				shutil.rmtree(self.localProjectDir)
		
		elif dtype == 'Depth':
			if not no_upload:
				self.uploadData(self.localSmoothDepthFile)
				self.uploadData(self.localRGBDepthVideo)
				self.uploadData(self.localRawDepthFile)
				self.uploadData(self.localInterpDepthFile)
				self.uploadData(self.localDepthLogfile)
				self.uploadData(self.localDailyDepthSummaryFigure)
				self.uploadData(self.localHourlyDepthSummaryFigure)

				#self.uploadData(self.localPaceDir)
			if delete:
				shutil.rmtree(self.localProjectDir)

		elif dtype == 'Cluster':
			if not no_upload:
				self.uploadData(self.localTroubleshootingDir)
				#self.uploadData(self.localPaceDir)

				if videoIndex is None:
					videos = list(range(len(self.lp.movies)))
				else:
					videos = [videoIndex]
				for videoIndex in videos:
					videoObj = self.returnVideoObject(videoIndex)
					self.uploadData(videoObj.localAllClipsDir, tarred = True)
					self.uploadData(videoObj.localManualLabelClipsDir, tarred = True)
					self.uploadData(videoObj.localManualLabelFramesDir, tarred = True)
					self.uploadData(videoObj.localLogfile)
			if delete:
				shutil.rmtree(self.localProjectDir)

		elif dtype == 'ClusterClassification':
			if not no_upload:
				self.uploadData(self.localAllLabeledClustersFile)
				self.uploadData(self.localClusterClassificationLogfile)
			if delete:
				shutil.rmtree(self.localProjectDir)
				try:
					shutil.rmtree(self.local3DModelDir)
				except AttributeError:
					pass

		elif dtype == 'TrackFish':
			if not no_upload:
				self.uploadData(self.localAllFishTracksFile)
				self.uploadData(self.localAllFishDetectionsFile)

			if delete:
				shutil.rmtree(self.localProjectDir)
				#os.remove(self.localYolov5WeightsFile)

		elif dtype == 'Train3DResnet':
			if not no_upload:
				self.uploadData(self.local3DModelDir)
			if delete:
				shutil.rmtree(self.localTempDir)
				shutil.rmtree(self.local3DModelTempDir)
				shutil.rmtree(self.localAnnotationDir)

		elif dtype == 'ManualLabelVideos':
			if not no_upload:
				self.uploadAndMerge(self.localNewLabeledVideosFile, self.localLabeledClipsFile, ID = 'LID')
				self.uploadAndMerge(self.localNewLabeledClipsDir, self.localLabeledClipsProjectDir, tarred = True)

			if delete:
				shutil.rmtree(self.localProjectDir)


		elif dtype == 'ManualLabelFrames':
			if not no_upload:
				self.uploadAndMerge(self.localNewLabeledFramesFile, self.localBoxedFishFile, ID = 'LID')
				self.uploadAndMerge(self.localNewLabeledFramesDir, self.localLabeledFramesProjectDir, tarred = True)
			if delete:
				shutil.rmtree(self.localProjectDir)

		elif dtype == 'Summary':
			self.uploadData(self.localSummaryDir)

		else:
			raise KeyError('Unknown key: ' + dtype)


	def returnVideoObject(self, index):
		self._createParameters()

		videoObj = self.lp.movies[index]
		videoObj.localVideoFile = self.localProjectDir + videoObj.mp4_file
		videoObj.localh264File = self.localProjectDir + videoObj.h264_file
		videoObj.localHMMFile = self.localTroubleshootingDir + videoObj.baseName + '.hmm'
		videoObj.localRawCoordsFile = self.localTroubleshootingDir + videoObj.baseName + '_rawCoords.npy'
		videoObj.localLabeledCoordsFile = self.localTroubleshootingDir + videoObj.baseName + '_labeledCoords.npy'
		videoObj.localLabeledClustersFile = self.localTroubleshootingDir + videoObj.baseName + '_labeledClusters.csv'
		videoObj.localFishDetectionsFile = self.localTroubleshootingDir + videoObj.baseName + '_fishDetections.csv'
		videoObj.localFishTracksFile = self.localTroubleshootingDir + videoObj.baseName + '_fishTracks.csv'

		videoObj.localAllClipsDir = self.localAllClipsDir + videoObj.baseName + '/'
		videoObj.localManualLabelClipsDir = self.localManualLabelClipsDir + videoObj.baseName + '/'
		videoObj.localManualLabelFramesDir = self.localManualLabelFramesDir + videoObj.baseName + '/'
		videoObj.localAllClipsPrefix = self.localAllClipsDir + self.lp.projectID + '_' + videoObj.baseName
		videoObj.localManualLabelClipsPrefix = self.localManualLabelClipsDir + self.lp.projectID + '_' + videoObj.baseName
		videoObj.localIntensityFile = self.localSummaryDir + videoObj.baseName + '_intensity.pdf'
		videoObj.localTempDir = self.localTempDir + videoObj.baseName + '/'
		videoObj.nManualLabelClips = int(self.nManualLabelClips/len(self.lp.movies))
		videoObj.nManualLabelFrames = int(self.nManualLabelFrames/len(self.lp.movies))
		videoObj.localLogfile = self.localLogfileDir + 'ClusterLog_' + str(index) + '.txt'

		self.createDirectory(videoObj.localTempDir)

		return videoObj

	def _createParameters(self):

		# Depth related parameters
		self.hourlyDepthThreshold = 0.2  # cm
		self.dailyDepthThreshold = 0.4  # cm
		self.totalDepthThreshold = 1.0  # cm

		# Cluster related parameters
		self.hourlyClusterThreshold = 0.6  # events/cm^2
		self.dailyClusterThreshold = 1.2  # events/cm^2
		self.totalClusterThreshold = 3.0  # events/cm^2

		# Parameters related to both depth and cluster analysis
		self.hourlyMinPixels = 1000
		self.dailyMinPixels = 1000
		self.totalMinPixels = 1000
		self.pixelLength = 0.1030168618  # cm / pixel
		self.bowerIndexFraction = 0.1

		# Video related parameters
		self.lightsOnTime = 8
		self.lightsOffTime = 18

		# DB Scan related parameters
		self.minMagnitude = 0
		self.treeR = 22 
		self.leafNum = 190 
		self.neighborR = 22
		self.timeScale = 10
		self.eps = 18
		self.minPts = 90 
		self.delta = 1.0 # Batches to calculate clusters

		# Clip creation parameters
		self.nManualLabelClips = 1200
		self.delta_xy = 100
		self.delta_t = 60
		self.smallLimit = 500

		# Manual Label Frame 
		self.nManualLabelFrames = 500

	def _identifyPiDirectory(self):
		writableDirs = []
		mounted_dir = '/media/pi/'
		try:
			possibleDirs = os.listdir(mounted_dir)
		except FileNotFoundError:
			return

		for d in possibleDirs:

			try:
				with open(mounted_dir + d + '/temp.txt', 'w') as f:
					print('Test', file = f)
				with open(mounted_dir + d + '/temp.txt', 'r') as f:
					for line in f:
						if 'Test' in line:
							writableDirs.append(d)
			except:
				pass
			try:
				os.remove(mounted_dir + d + '/temp.txt')
			except FileNotFoundError:
				continue
		
		if len(writableDirs) == 1:
			self.localMasterDir = mounted_dir + d + '/CichlidAnalyzer/'
			self.system = 'pi'
		elif len(writableDirs) == 0:
			raise Exception('No writable drives in /media/pi/')
		else:
			raise Exception('Multiple writable drives in /media/pi/. Options are: ' + str(writableDirs))

	def createDirectory(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def downloadData(self, local_data, tarred = False, tarred_subdirs = False, allow_errors=False, quiet=False):
		if local_data is None:
			return
		relative_name = local_data.rstrip('/').split('/')[-1] + '.tar' if tarred else local_data.rstrip('/').split('/')[-1]
		local_path = local_data.split(local_data.rstrip('/').split('/')[-1])[0]
		cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

		cloud_objects = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8').stdout.split()

		if relative_name + '/' in cloud_objects: #directory
			output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path + relative_name], capture_output = True, encoding = 'utf-8')
		elif relative_name in cloud_objects: #file
			output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path], capture_output = True, encoding = 'utf-8')
		else:
			if allow_errors:
				if not quiet:
					print('Warning: Cannot find {}. Continuing'.format(cloud_path + relative_name))
				else:
					pass
			else:
				raise FileNotFoundError('Cant find file for download: ' + cloud_path + relative_name)

		if not os.path.exists(local_path + relative_name):
			if allow_errors:
				if not quiet:
					print('Warning. Cannot download {}. Continuing'.format(local_path + relative_name))
				else:
					pass
			else:
				raise FileNotFoundError('Error downloading: ' + local_path + relative_name)

		if tarred:
			# Untar directory
			output = subprocess.run(['tar', '-xvf', local_path + relative_name, '-C', local_path], capture_output = True, encoding = 'utf-8')
			output = subprocess.run(['rm', '-f', local_path + relative_name], capture_output = True, encoding = 'utf-8')

		if tarred_subdirs:
			for d in [x for x in os.listdir(local_data) if '.tar' in x]:
				output = subprocess.run(['tar', '-xvf', local_data + d, '-C', local_data, '--strip-components', '1'], capture_output = True, encoding = 'utf-8')
				os.remove(local_data + d)

	def uploadData(self, local_data, tarred = False):

		attempt = 1
		while True:
			relative_name = local_data.rstrip('/').split('/')[-1]
			local_path = local_data.split(relative_name)[0]
			cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

			if tarred:
				output = subprocess.run(['tar', '-cvf', local_path + relative_name + '.tar', '-C', local_path, relative_name], capture_output = True, encoding = 'utf-8')
				if output.returncode != 0:
					print(output.stderr)
					if attempt < 3:
						attempt += 1
						continue
					raise Exception('Error in tarring ' + local_data)
				relative_name += '.tar'

			if os.path.isdir(local_path + relative_name):
				output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path + relative_name], capture_output = True, encoding = 'utf-8')
				#subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path + relative_name], check = True) #Troubleshooting directory will have depth data in it when you upload the cluster data

			elif os.path.isfile(local_path + relative_name):
				#print(['rclone', 'copy', local_path + relative_name, cloud_path])
				output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path], capture_output = True, encoding = 'utf-8')
				#output = subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path], check = True, capture_output = True, encoding = 'utf-8')
			else:
				raise Exception(local_data + ' does not exist for upload')

			if output.returncode != 0:
				if attempt < 3:
					attempt += 1
					continue
				raise Exception('Error in uploading file: ' + output.stderr)
			else:
				return

	def uploadAndMerge(self, local_data, master_file, tarred = False, ID = False):
		if os.path.isfile(local_data):
			#We are merging two crv files
			self.downloadData(master_file)
			import pandas as pd
			if ID:
				old_dt = pd.read_csv(master_file, index_col = ID)
				new_dt = pd.read_csv(local_data, index_col = ID)
				old_dt = old_dt.append(new_dt)
				old_dt.index.name = ID
			else:
				old_dt = pd.read_csv(master_file)
				new_dt = pd.read_csv(local_data)
				old_dt = old_dt.append(new_dt)
			
			old_dt.to_csv(master_file, sep = ',')
			self.uploadData(master_file)
		else:
			#We are merging two tarred directories
			try:		
				self.downloadData(master_file, tarred = True)
			except FileNotFoundError:
				self.createDirectory(master_file)
			for nfile in os.listdir(local_data):
				subprocess.run(['mv', local_data + nfile, master_file])
			self.uploadData(master_file, tarred = True)

	def checkFileExists(self, local_data):
		relative_name = local_data.rstrip('/').split('/')[-1]
		local_path = local_data.split(relative_name)[0]
		cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

		output = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8')
		remotefiles = [x.rstrip('/') for x in output.stdout.split('\n')]

		if relative_name in remotefiles:
			return True
		else:
			return False

	def deleteCloudData(self, local_data):
		if self.checkFileExists(local_data):
			cloud_path = local_data.replace(self.localMasterDir, self.cloudMasterDir)
			output = subprocess.run(['rclone', 'purge', cloud_path], capture_output = True, encoding = 'utf-8')
			if self.checkFileExists(local_data):
				pdb.set_trace()
			else:
				return

