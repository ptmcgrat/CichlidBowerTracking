
import subprocess, pickle, os, shutil, pdb, scipy, datetime
from skimage import io
import pandas as pd
import re

class ThreeDClassifierPreparer:
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndices):
		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.videoIndices = videoIndices
	
	def downloadData(self):
		self.fileManager.downloadData(self.fileManager.localModelCommandsFile)
		self.fileManager.downloadData(self.fileManager.localVideoModelFile)
		self.fileManager.downloadData(self.fileManager.localModelDataBreakdown)
		self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			self.fileManager.downloadData(videoObj.localAllClipsDir, tarred = True)
			self.fileManager.downloadData(videoObj.localLabeledClustersFile)

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localVideoModelFile) # Model
		assert os.path.exists(self.fileManager.localModelDataBreakdown) # Classes
		assert os.path.exists(self.fileManager.localModelCommandsFile) # Commands

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localAllClipsDir)
			assert os.path.exists(videoObj.localLabeledClustersFile)

	def predictLabels(self):
		if not os.path.isdir('CAC_MD'):
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionClassification', '--branch', 'mcgrath_dev', '--single-branch','CAC_MD'], capture_output = True)

		#command = "source activate CichlidActionClassification; " + ' ' .join(command)
		#command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
		os.chdir('CAC_MD')			
		subprocess.run(['git','pull'], capture_output = True)
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			data = {}
			data['ClipName'] = [x for x in os.listdir(videoObj.localAllClipsDir) if '.mp4' in x]
			data['ProjectID'] = self.fileManager.projectID
			data['ManualLabel'] = 'x' # Doesn't matter what you put in here
			data['AnalysisID'] = self.fileManager.analysisID
			dt = pd.DataFrame(data)
			dt.to_csv(self.fileManager.localVideoProjectsFile)

			# Run command
			command = ['python3', 'ClassifyVideos.py']
			command.extend(['--Clips_directory', videoObj.localAllClipsDir])
			command.extend(['--ML_labels', self.fileManager.localVideoProjectsFile])
			command.extend(['--Temp_directory', videoObj.localAllClipsDir])
			command.extend(['--Results_directory', videoObj.localAllClipsDir])

			command.extend(['--CommandsLog', self.fileManager.localModelCommandsFile])
			command.extend(['--JSONLog', self.fileManager.localModelDataBreakdown])
			command.extend(['--CondaLog', videoObj.localClassifyLogfile])
			command.extend(['--Trained_model', self.fileManager.localVideoModelFile])

			command.extend(['--Output_file', videoObj.localAllClipsDir + 'output.csv'])
			print(' '.join(command))

			subprocess.run(command)
			pdb.set_trace()
		

	def createSummaryFile(self):
		
		# for videoIndex, video in enumerate(self.fileManager.lp.movies):
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			new_dt = pd.read_csv(videoObj.localLabeledClustersFile)
			pred_dt = pd.read_csv(videoObj.localAllClipsDir + 'output.csv')
			temp_dt = pd.merge(new_dt, pred_dt, on = 'ClipName', how = 'left')
			try:
				# c_dt = c_dt.append(new_dt)
				c_dt = pd.concat([c_dt, temp_dt], ignore_index=True)
			except NameError:
				c_dt = temp_dt
		c_dt['ProjectID'] = self.fileManager.projectID
		c_dt = c_dt[['ProjectID','VideoID','ClipName','t','X','Y','N','InFrame','ClipCreated','TimeStamp','Prediction','Probability']]
		c_dt.to_csv(self.fileManager.localAllLabeledClustersFile)

	def uploadData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localAllLabeledClustersFile)
		if delete:
			shutil.rmtree(self.fileManager.localProjectDir)

