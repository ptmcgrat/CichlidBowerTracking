import subprocess, os, pdb


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
		self.workers = workers
		self.videoIndex = videoIndex


	def validateInputData(self):
		
		assert os.path.exists(self.videoObj.localVideoFile)
		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localAllClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelClipsDir)
		assert os.path.exists(self.fileManager.localManualLabelFramesDir)


		"""self.uploads = [(self.fileManager.localTroubleshootingDir, self.fileManager.cloudTroubleshootingDir, '0'), 
						(self.fileManager.localAnalysisDir, self.fileManager.cloudAnalysisDir, '0'),
						(self.fileManager.localAllClipsDir, self.fileManager.cloudMasterDir, '1'),
						(self.fileManager.localManualLabelClipsDir, self.fileManager.cloudMasterDir, '1'),
						(self.fileManager.localManualLabelFramesDir, self.fileManager.cloudMasterDir, '1'),
						(self.fileManager.localManualLabelFramesDir[:-1] + '_pngs', self.fileManager.cloudMasterDir[:-1] + '_pngs', '1')
						]"""

	def runClusterAnalysis(self):
		args = ['python3', 'CichlidActionDetection/VideoFocus.py']
		args.extend(['--Movie_file', self.videoObj.localVideoFile])
		args.extend(['--Num_workers', self.workers])
		args.extend(['--Log', self.videoObj.localHMMFile + '.log'])
		args.extend(['--HMM_temp_directory', self.videoObj.localTempDir])
		args.extend(['--HMM_filename', self.videoObj.localHMMFile])
		args.extend(['--HMM_transition_filename', self.videoObj.localRawCoordsFile])
		args.extend(['--Cl_labeled_transition_filename', self.videoObj.localLabeledCoordsFile])
		args.extend(['--Cl_labeled_cluster_filename', self.videoObj.localLabeledClustersFile])
		args.extend(['--Cl_videos_directory', self.fileManager.localAllClipsDir])
		args.extend(['--ML_frames_directory', self.fileManager.localManualLabelFramesDir])
		args.extend(['--ML_videos_directory', self.fileManager.localManualLabelClipsDir])
		args.extend(['--Video_start_time', str(self.videoObj.startTime)])
		args.extend(['--VideoID', self.fileManager.lp.movies[0].baseName])

		pdb.set_trace()

		subprocess.run(args)



