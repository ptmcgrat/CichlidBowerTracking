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

	def __init__(self, fileManager):
		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.videos =[]
	

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localAllClipsDir) # Clips
		assert os.path.exists(self.fileManager.localVideoModelFile) # Model
		assert os.path.exists(self.fileManager.localVideoClassesFile) # Classes
		assert os.path.exists(self.fileManager.localModelCommandsFile) # Commands


		# df = pd.read_csv(self.fileManager.localSummaryFile)
		
		# videos = list(range(len(self.fileManager.lp.movies)))
		self.fileManager.s_dt.columns = [c.replace(' ', '_') for c in self.fileManager.s_dt.columns]
		# pdb.set_trace()
		row =self.fileManager.s_dt.loc[self.fileManager.projectID]
		videos = row['VideoIDs_new'].split(': ')[1].split(',')
		self.videos = list(map(int, videos))
		# videos = self.fileManager.s_dt['Video']
		
		for videoIndex in self.videos:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			# pdb.set_trace()
			assert os.path.exists(videoObj.localLabeledClustersFile)


	def predictLabels(self):


		# Create mapping from videos to projectID

		with open(self.fileManager.localVideoProjectsFile, 'w') as f:
			print('Location,ManualLabel,ProjectID,ClipName,MeanID', file = f)

			for videofile in [x for x in os.listdir(self.fileManager.localAllClipsDir) if '.mp4' in x]:
				clipname = self.fileManager.projectID+'__'+videofile.replace(".mp4", "")
				match = re.match(r'(.*?)__(\d+_vid)', clipname)
				mean_ID = match.group(1) + ':' + match.group(2)
				print(videofile + ',,' + self.fileManager.projectID+','+clipname+','+mean_ID, file = f)
				# pdb.set_trace()

		# Run command
		command = ['python3', 'ClassifyVideos.py']
		command.extend(['--Input_videos_directory', self.fileManager.localAllClipsDir])
		command.extend(['--Videos_to_project_file', self.fileManager.localVideoProjectsFile])
		command.extend(['--Trained_model', self.fileManager.localVideoModelFile])
		command.extend(['--Training_options', self.fileManager.localModelCommandsFile])
		command.extend(['--Trained_categories', self.fileManager.localVideoClassesFile])
		command.extend(['--Temporary_output_directory', self.fileManager.localTempClassifierDir])
		command.extend(['--Output_file', self.fileManager.localAnalysisDir + 'output.csv'])

		print(' '.join(command))


		if not os.path.isdir('/home/pkolipaka3@ad.gatech.edu/Desktop/Prerana/CichlidActionClassification'):
			subprocess.run(['git', 'clone', 'https://github.com/ptmcgrat/CichlidActionClassification.git'])

		#command = "source activate CichlidActionClassification; " + ' ' .join(command)
		# command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
		command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidClusterData; " + ' '.join(command) # change the environment name
		os.chdir('/home/pkolipaka3@ad.gatech.edu/Desktop/Prerana/CichlidActionClassification') # change the path to the path where this repo is cloned
		# subprocess.run(['git', 'pull'])
		subprocess.run('bash -c \"' + command + '\"', shell = True)
		os.chdir('..')
		shutil.copy(os.path.join(self.fileManager.local3DModelDir,'train.log'), self.fileManager.localClusterClassificationLogfile)

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
