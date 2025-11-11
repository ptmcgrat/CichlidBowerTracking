
import subprocess, os, pdb, shutil
import pandas as pd

class ThreeDModelPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager

	def downloadProjectData(self):		
		self.fileManager.createDirectory(self.fileManager.local3DModelDir)
		self.fileManager.createDirectory(self.fileManager.local3DModelTempDir)

		self.fileManager.downloadData(self.fileManager.localLabeledClipsDir, tarred_subdirs = True)
		self.fileManager.downloadData(self.fileManager.localLabeledClipsFile)

	def validateInputData(self):
		
		assert os.path.exists(self.fileManager.localLabeledClipsDir)
		assert os.path.exists(self.fileManager.localLabeledClipsFile)
		assert os.path.exists(self.fileManager.local3DModelDir)
		assert os.path.exists(self.fileManager.local3DModelTempDir)

	def create3DModel(self):
		
		# Filter out annotated videos so they only include projects requested
		dt = pd.read_csv(self.fileManager.localLabeledClipsFile, index_col = 0)
		dt['ProjectID'] = dt.ClipName.str.split('__').str[0]
		#if self.projects is not None:
		#	dt.loc[~dt.ProjectID.isin(self.projects),'Dataset'] = 'Validate'
		dt['ClipName'] = dt.ClipName + '.mp4'
		#dt = dt.rename(columns = {'ClipName':'VideoFile', 'ManualLabel':'Label'})
		dt = dt[(dt.AnalysisID != 'Single_nuc_1')]
		dt.to_csv(self.fileManager.localVideoProjectsFile)

		command = ['python3', 'TrainModel.py']
		command.extend(['--Input_videos_directory', self.fileManager.localLabeledClipsDir])
		command.extend(['--ML_labels', self.fileManager.localVideoProjectsFile])
		command.extend(['--Results_directory', self.fileManager.local3DModelDir])
		command.extend(['--Temporary_clips_directory', self.fileManager.local3DModelTempDir])
		
		command.extend(['--Log', self.fileManager.localModelCondaVersionsFile])
		
		command.extend(['--gpu', str(0)])
		

		if not os.path.isdir('VideoClassifier'):
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionClassification'])

		#command = "source activate CichlidActionClassification; " + ' ' .join(command)
		#command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
		os.chdir('CichlidActionClassification')
		subprocess.run(['git', 'pull'])
		subprocess.run(command)
		os.chdir('..')
		
		with open(os.path.join(self.fileManager.local3DModelDir,'val.log')) as f:
			print('Epoch\tAccuracy')
			for line in f:
				try:
					if int(line.split()[0]) % 5 == 0:
						print(line.split()[0] + '\t' + line.rstrip().split()[-1])
				except ValueError:
					continue
			epoch = 1
			while epoch % 5 != 0:
				try:
					epoch = int(input('Choose epoch to use'))
				except ValueError:
					continue
			# Move files
			#shutil.copy(os.path.join(self.fileManager.local3DModelDir,'val.log'), self.fileManager.local3DModelDir)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'save_' + str(epoch) + '.pth'), self.fileManager.localVideoModelFile)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'epoch_' + str(epoch) + '_confusion_matrix.csv'), self.fileManager.localVideoLabels)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'project_data_' + str(epoch) + '.csv'), self.fileManager.local3DModelDir + 'ProjectAccuracy.csv')
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'TrainingLog.txt'), self.fileManager.localModelCommandsFile)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'VideoSplit.csv'), self.fileManager.localVideoProjectsFile)
			#shutil.copy(os.path.join(self.fileManager.local3DModelDir,'MissingVideos.csv'), self.fileManager.local3DModelDir)
			#shutil.copy(os.path.join(self.fileManager.local3DModelDir,'classInd.txt'), self.fileManager.local3DModelDir)
