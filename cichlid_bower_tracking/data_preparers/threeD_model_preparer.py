
import subprocess, os, pdb, shutil, json
import pandas as pd

class ThreeDModelPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, exclude):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.exclude = exclude

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
		if self.exclude is not None:
			dt = dt[~dt.AnalysisID.isin(self.exclude)]
		dt = dt[(dt.ManualLabel != 'u') & (dt.ManualLabel != 'x')]
		dt.to_csv(self.fileManager.localVideoProjectsFile)

		command = ['python3', 'TrainModel.py']
		command.extend(['--Clips_directory', self.fileManager.localLabeledClipsDir])
		command.extend(['--Temp_directory', self.fileManager.local3DModelTempDir])
		command.extend(['--ML_labels', self.fileManager.localVideoProjectsFile])
		command.extend(['--Results_directory', self.fileManager.local3DModelDir])
		command.extend(['--DataSummaryLog', self.fileManager.localModelDataSummary])
		command.extend(['--CommandsLog', self.fileManager.localModelCommandsFile])
		command.extend(['--JSONLog', self.fileManager.localModelDataBreakdown])
		command.extend(['--CondaLog', self.fileManager.localModelCondaVersionsFile])		
		command.extend(['--gpu', str(0)])
		

		if not os.path.isdir('CAC_MD'):
			subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionClassification', '--branch', 'mcgrath_dev', '--single-branch','CAC_MD'], capture_output = True)

		#command = "source activate CichlidActionClassification; " + ' ' .join(command)
		#command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
		os.chdir('CAC_MD')
		subprocess.run(['git', 'pull'], capture_output = True)
		#subprocess.run(command)
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
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'save_' + str(epoch) + '.pth'), self.fileManager.localVideoModelFile)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'epoch_' + str(epoch) + '_confusion_matrix.csv'), self.fileManager.localModelConfusionFile)
			shutil.copy(os.path.join(self.fileManager.local3DModelDir,'epoch_' + str(epoch) + '_accuracy.csv'), self.fileManager.localModelProjectAccuracy)
			#shutil.copy(os.path.join(self.fileManager.local3DModelDir,'source.json'), self.fileManager.localModelDataBreakdown)



	def uploadData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localModelDataSummary)
		self.fileManager.uploadData(self.fileManager.localModelCondaVersionsFile)
		self.fileManager.uploadData(self.fileManager.localModelCommandsFile)
		self.fileManager.uploadData(self.fileManager.localVideoModelFile)
		self.fileManager.uploadData(self.fileManager.localModelDataBreakdown)
		self.fileManager.uploadData(self.fileManager.localModelConfusionFile)
		self.fileManager.uploadData(self.fileManager.localModelProjectAccuracy)
		
		if delete:
			shutil.rmtree(self.fileManager.local3DModelDir)

  
