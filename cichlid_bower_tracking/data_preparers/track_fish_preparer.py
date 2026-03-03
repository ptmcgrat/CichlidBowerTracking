import pdb, subprocess, os, csv, datetime, shutil

class TrackFishPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndices, mode):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.videoIndices = videoIndices
		self.mode = mode

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			self.fileManager.downloadData(videoObj.localVideoFile)

		self.fileManager.downloadData(self.fileManager.localYOLOModelDir)
		self.fileManager.downloadData(self.fileManager.localPoseModelDir)

		#self.createLogFile()

	def validateInputData(self):

		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localYOLOModelFile)
		assert os.path.exists(self.fileManager.localPoseModelFile)

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localVideoFile)


	def createLogFile(self):
		
		# with open(self.fileManager.localClusterLogfile,'w') as f:
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)

			with open(videoObj.localYOLOLogfile,'w') as f:
				print('GitBranch: ' + self.fileManager.branch_name, file = f)
				print('Username: ' + os.getenv('USER'), file = f)
				print('Nodename: ' + os.uname().nodename, file = f)
				print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)
				output = subprocess.run(['conda','list'], capture_output = True)
				print(output.stdout.decode('utf-8'), file = f)

	def runYOLOAnalysis(self):

		processes = []
		
		if self.mode == 'TrackFish':
			input_model = self.fileManager.localYOLOModelFile
			output_file = videoObj.localFishDetectionsFile
		elif self.mode == 'PoseFish':
			input_model = self.fileManager.localPoseModelFile
			output_file = videoObj.localFishPoseFile

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localVideoFile)
			processes.append(subprocess.Popen(['python3', 'unit_scripts/track_video.py', videoObj.localVideoFile, output_file, input_model]))
		
		for p1 in processes:
			p1.communicate()

	def uploadProjectData(self, delete = True):
		
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
		
			if self.mode == 'TrainFish':
				self.fileManager.uploadData(videoObj.localFishDetectionsFile)
			elif self.mode == 'PoseFish':
				self.fileManager.uploadData(videoObj.localFishPoseFile)
			
			#self.fileManager.uploadData(videoObj.localYOLOLogfile)

			if delete:
				os.remove(videoObj.localVideoFile)
				os.remove(videoObj.localFishDetectionsFile)
