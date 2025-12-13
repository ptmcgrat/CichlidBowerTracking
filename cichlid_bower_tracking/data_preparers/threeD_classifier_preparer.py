
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

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			data = {}
			data['VideoFile'] = [x for x in os.listdir(videoObj.localAllClipsDir) if '.mp4' in x]
			data['ProjectID'] = self.fileManager.projectID
			data['Label'] = 'x' # Doesn't matter what you put in here
			dt = pd.DataFrame(data)
			dt.to_csv(self.fileManager.localVideoProjectsFile)

			# Run command
			command = ['python3', 'ClassifyVideos.py']
			command.extend(['--Input_videos_directory', videoObj.localAllClipsDir])
			command.extend(['--Results_directory', videoObj.localAllClipsDir])
			command.extend(['--Temporary_clips_directory', videoObj.localAllClipsDir])

			command.extend(['--Videos_to_project_file', self.fileManager.localVideoProjectsFile])
			command.extend(['--Trained_model', self.fileManager.localVideoModelFile])
			command.extend(['--Training_options', self.fileManager.localModelCommandsFile])
			command.extend(['--Trained_categories', self.fileManager.localModelDataBreakdown])
			command.extend(['--Temporary_output_directory', videoObj.localAllClipsDir])
			command.extend(['--Output_file', videoObj.localAllClipsDir + 'output.csv'])

		print(' '.join(command))

		if not os.path.isdir('CichlidActionClassification'):
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionClassification'])

		#command = "source activate CichlidActionClassification; " + ' ' .join(command)
		#command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
		os.chdir('CichlidActionClassification')
		subprocess.run(['git', 'pull'])
		subprocess.run(command)
		os.chdir('..')
		#shutil.copy(os.path.join(self.fileManager.local3DModelDir,'train.log'), self.fileManager.localClusterClassificationLogfile)

	def createSummaryFile(self):
		
		# for videoIndex, video in enumerate(self.fileManager.lp.movies):
		for videoIndex in self.videos:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			new_dt = pd.read_csv(videoObj.localLabeledClustersFile)
			try:
				# c_dt = c_dt.append(new_dt)
				c_dt = pd.concat([c_dt, new_dt], ignore_index=True)
			except NameError:
				c_dt = new_dt
		# pdb.set_trace()
		pred_dt = pd.read_csv(os.path.join(self.fileManager.localAnalysisDir,'output.csv'), index_col = 0)
		# pdb.set_trace()
		pred_dt['ClipName'] = pred_dt.index.str.replace('.mp4','')
		# pdb.set_trace()
		out_dt = pd.merge(c_dt, pred_dt[['ClipName','predicted_label']], on='ClipName', how = 'left')
		# out_dt['confidence'] = out_dt['predicted_label'].map(lambda label: pred_dt[label].values[0] if label in pred_dt.columns else None)
		# pdb.set_trace()
		# out_dt['confidence'] = out_dt.apply(lambda row: row[row['predicted_label']] if row['predicted_label'] in pred_dt.columns else None, axis=1)
		out_dt['confidence'] = out_dt.apply(lambda row: pred_dt.loc[row['ClipName'], row['predicted_label']] if row['ClipName'] in pred_dt.index and row['predicted_label'] in pred_dt.columns else None, axis=1)
		# pred_dt.to_csv(os.path.join(self.fileManager.localAnalysisDir,'output.csv'))
		# pdb.set_trace()
		out_dt['modelID'] = self.fileManager.modelID
		# pdb.set_trace()
		out_dt.to_csv(self.fileManager.localAllLabeledClustersFile)
		# pdb.set_trace()
