import subprocess, os, pdb, sys
import datetime
import numpy as np
import scipy
import shutil

class ClusterPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndex, workers):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.videoObj = self.fileManager.returnVideoObject(videoIndex)
		self.workers = workers # Fix this
		self.videoIndex = videoIndex
		self.createLogFile()

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
		self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
		self.fileManager.createDirectory(self.fileManager.localTempDir)
		self.fileManager.createDirectory(self.fileManager.localAllClipsDir)
		self.fileManager.createDirectory(self.fileManager.localManualLabelClipsDir)
		self.fileManager.createDirectory(self.fileManager.localManualLabelFramesDir)
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)

		self.fileManager.downloadData(self.fileManager.localVideoCropFile)
		self.fileManager.downloadData(self.fileManager.localTransMFile)
		self.fileManager.downloadData(self.fileManager.localLogfile)
		# self.fileManager.downloadData(self.fileManager.localVideoFile)
		try:
			self.fileManager.downloadData(self.videoObj.localVideoFile)
		except FileNotFoundError:
			print(self.videoObj.localVideoFile + ' not found on cloud. Trying h264 file')
			self.fileManager.downloadData(self.videoObj.localh264File)
			command = ['ffmpeg', '-r', str(self.videoObj.framerate), '-i', self.videoObj.localh264File, '-threads', str(self.workers), '-c:v', 'copy', '-r', str(self.videoObj.framerate), self.videoObj.localVideoFile]
			ffmpeg_output = subprocess.run(command, capture_output = True)
			if ffmpeg_output.returncode != 0:
				print(ffmpeg_output.stderr.decode('utf-8'))
			assert os.path.isfile(self.videoObj.localVideoFile)
			assert os.path.getsize(self.videoObj.localVideoFile) > os.path.getsize(self.videoObj.localh264File)
			#self.fileManager.uploadData(self.videoObj.localVideoFile)
			subprocess.run(['rm', '-f', self.videoObj.localh264File])

	def validateInputData(self):

		assert os.path.exists(self.videoObj.localVideoFile)
		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localAllClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelFramesDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localVideoCropFile)
		assert os.path.exists(self.fileManager.localTransMFile)

	def createLogFile(self):
		
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		# with open(self.fileManager.localClusterLogfile,'w') as f:
		with open(self.videoObj.localLogfile,'w') as f:
			print('GitBranch: ' + self.fileManager.branch_name, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)
			output = subprocess.run(['conda','list'], capture_output = True)
			print(output.stdout.decode('utf-8'), file = f)

	def runClusterAnalysis(self):

		command = ['python3', 'VideoFocus.py']
		command.extend(['--Movie_file', self.videoObj.localVideoFile])
		command.extend(['--Video_framerate', str(self.videoObj.framerate)])
		command.extend(['--Num_workers', str(self.workers)])
		command.extend(['--Log', self.videoObj.localLogfile])
		command.extend(['--HMM_temp_directory', self.videoObj.localTempDir])
		command.extend(['--HMM_filename', self.videoObj.localHMMFile])
		command.extend(['--HMM_transition_filename', self.videoObj.localRawCoordsFile])
		command.extend(['--Cl_labeled_transition_filename', self.videoObj.localLabeledCoordsFile])
		command.extend(['--Cl_labeled_cluster_filename', self.videoObj.localLabeledClustersFile])
		command.extend(['--Cl_videos_directory', self.videoObj.localAllClipsDir])
		command.extend(['--ML_frames_directory', self.videoObj.localManualLabelFramesDir])
		command.extend(['--ML_videos_directory', self.videoObj.localManualLabelClipsDir])
		command.extend(['--Video_start_time', str(self.videoObj.startTime)])
		command.extend(['--VideoID', self.videoObj.baseName])

		if not os.path.isdir('CichlidActionDetection'):
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionDetection'], capture_output = True)

		print('Running CichlidActionDetection on: ' + self.videoObj.localVideoFile.replace(self.fileManager.localMasterDir, ''))
		os.chdir('CichlidActionDetection')
		subprocess.run(['git', 'pull'], capture_output = True)
		subprocess.run(command)
		os.chdir('..')

	def addCropAndDepthCoordinates(self):
		transM = np.load(self.fileManager.localTransMFile)

		clusterData = pd.read_csv(videoObj.localLabeledClustersFile)
		clusterData['X_depth'] = clusterData.apply(
			lambda row: (transM[0][0] * row.Y + transM[0][1] * row.X + transM[0][2]) / (
					transM[2][0] * row.Y + transM[2][1] * row.X + transM[2][2]), axis=1)
		clusterData['Y_depth'] = self.clusterData.apply(
			lambda row: (transM[1][0] * row.Y + transM[1][1] * row.X + transM[1][2]) / (
					transM[2][0] * row.Y + transM[2][1] * row.X + transM[2][2]), axis=1)
		with open(self.fileManager.localVideoCropFile) as f:
			for line in f:
				video_crop_points = eval(line.rstrip())
		polygon = Polygon(video_crop_points)
		buffered_polygon = polygon.buffer(40, join_style=2)
		clusterData['InFrame'] = self.clusterData.apply(lambda row: buffered_polygon.contains(Point(row['Y'],row['X'])), axis = 1)
		clusterData.to_csv(videoObj.localLabeledClustersFile)

	def uploadProjectData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localTroubleshootingDir)
		self.fileManager.uploadData(self.videoObj.localAllClipsDir, tarred = True)
		self.fileManager.uploadData(self.videoObj.localManualLabelClipsDir, tarred = True)
		self.fileManager.uploadData(self.videoObj.localManualLabelFramesDir, tarred = True)
		self.fileManager.uploadData(self.videoObj.localLogfile)

		if delete:
			shutil.rmtree(self.fileManager.localProjectDir)
