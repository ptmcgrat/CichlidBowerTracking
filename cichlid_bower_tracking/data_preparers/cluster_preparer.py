import subprocess, os, pdb, sys
import datetime
import skvideo
import numpy as np
import scipy

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

		self.fileManager.downloadData(self.fileManager.localLogfile)
		# self.fileManager.downloadData(self.fileManager.localVideoFile)
		self.fileManager.downloadData(self.videoObj.localVideoFile)


	def validateInputData(self):

		assert os.path.exists(self.videoObj.localVideoFile)

		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localAllClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelFramesDir)
		assert os.path.exists(self.fileManager.localLogfileDir)

	def createLogFile(self):
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		# with open(self.fileManager.localClusterLogfile,'w') as f:
		with open(self.videoObj.localLogfile,'w') as f:
			print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
			print('NumpyVersion: ' + np.__version__, file = f)
			print('Scikit-VideoVersion: ' + skvideo.__version__, file = f)
			print('ScipyVersion: ' + scipy.__version__, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)


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
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionDetection'])

		os.chdir('CichlidActionDetection')
		subprocess.run(['git', 'pull'])
		subprocess.run(command)
		os.chdir('..')

	def uploadProjectData(self, delete = True):
		self.uploadData(self.localTroubleshootingDir)
		self.uploadData(self.videoObj.localAllClipsDir, tarred = True)
		self.uploadData(self.videoObj.localManualLabelClipsDir, tarred = True)
		self.uploadData(self.videoObj.localManualLabelFramesDir, tarred = True)
		self.uploadData(self.videoObj.localLogfile)

		if delete:
			shutil.rmtree(self.localProjectDir)
