from helper_modules.depth_analyzer import ClusterAnalyzer as CA
import subprocess, os, sys
import pdb, datetime, os, subprocess, argparse, random, cv2
import pandas as pd
import shutil

class EditVideosPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndices):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.videoIndices = videoIndices
		# 10 categories of annotation plus quit and skip commands
		self.commands = ['c','f','p','t','b','m','s','x','o','d','q','k','r']
		self.commands_help = "Type 'c': BuildScoop; 'f': FeedScoop; 'p': BuildSpit; 't': FeedSpit; 'b': BuildMultiple; 'm': FeedMultiple; s': Spawn; 'x': Reflection; 'o': FishOther; 'd': DropSand; 'q': quit; 'k': skip; 'r': redo"

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
		self.fileManager.createDirectory(self.fileManager.localEditVideosDir)
		self.fileManager.downloadData(self.fileManager.localVideoCropFile)
		self.fileManager.downloadData(self.fileManager.localTransMFile)

		self.fileManager.downloadData(self.fileManager.localAllLabeledClustersFile)
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			self.fileManager.downloadData(videoObj.localVideoFile)

	def validateInputData(self):
		
		assert os.path.exists(self.fileManager.localEditVideosDir)
		assert os.path.exists(self.fileManager.localLabeledClipsFile)
		assert os.path.exists(self.fileManager.localVideoCropFile)
		assert os.path.exists(self.fileManager.localTransMFile)

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localVideoFile)
	
	def uploadProjectData(self, delete = True):

		self.fileManager.uploadData(self.fileManager.localEditVideosDir)

		if delete:
			try:
				shutil.rmtree(self.fileManager.localProjectDir)
			except FileNotFoundError:
				pass

	def editVideos(self):
		self.cl_obj = CA(self.fileManager)
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			cap = cv2.VideoCapture(videoObj.localVideoFile)
			out_file = self.fileManager.localEditVideosDir + videoObj.baseName + '.mp4'
			outAll = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), videoObj.framerate, (videoObj.width, videoObj.height))
			for i in range(int(videoObj.framerate*60*10)):
				current_time = videoObj.startTime + datetime.timedelta(seconds = i/videoObj.framerate)
				out_dt = self.cl_obj.addClusterLabels(current_time, videoObj)
				if len(out_dt) > 0:
					pdb.set_trace()
				ret, frame = cap.read()
				if ret:
					for time, row in out_dt.iterrows():
						cv2.rectangle(frame, (row.x1, row.y1), (row.x2, row.y2), color=row.color, thickness=2)
						cv2.putText(frame, row.label, (row.x1, row.y1), cv2.FONT_HERSHEY_SIMPLEX, 1, row.color, 2)
					outAll.write(frame)
				else:
					print('VideoError: BadFrame')

			outAll.release()
			break
