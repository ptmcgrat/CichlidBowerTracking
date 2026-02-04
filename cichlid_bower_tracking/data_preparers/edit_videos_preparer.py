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

		self.fileManager.downloadData(self.fileManager.localAllLabeledClustersFile)
		self.fileManager.downloadData(self.fileManager.localAllFishTracksFile)
		self.fileManager.downloadData(self.fileManager.localAllTracksSummaryFile)
		self.fileManager.downloadData(self.fileManager.localVideoCropFile)
		self.fileManager.downloadData(self.fileManager.localTransMFile)

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			self.fileManager.downloadData(videoObj.localVideoFile)

	def validateInputData(self):
		
		assert os.path.exists(self.fileManager.localEditVideosDir)
		assert os.path.exists(self.fileManager.localLabeledClipsFile)

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
		t_dt = pd.read_csv(self.fileManager.localAllFishTracksFile)

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			cap = cv2.VideoCapture(videoObj.localVideoFile)
			out_file = self.fileManager.localEditVideosDir + videoObj.baseName + '.mp4'
			outAll = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), videoObj.framerate, (videoObj.width, videoObj.height))
			for i in range(int(videoObj.framerate*60*15)):
				current_time = videoObj.startTime + datetime.timedelta(seconds = i/videoObj.framerate)
				out_dt = self.cl_obj.addClusterLabels(current_time, videoObj)
				ret, frame = cap.read()
				sub_dt = t_dt[t_dt.FrameNum == i]
				if ret:
					for time, row in out_dt.iterrows():
						cv2.rectangle(frame, (row.x1, row.y1), (row.x2, row.y2), color=row.color, thickness=2)
						cv2.putText(frame, row.label, (row.x1, row.y1), cv2.FONT_HERSHEY_SIMPLEX, 1, row.color, 2)
					if len(sub_dt) != 0:
						for t_id, row in sub_dt.iterrows():
							if row.Sex == 'female':
								cv2.rectangle(frame, (int(row.X_center - row.Width/2), int(row.Y_center - row.Height/2)), (int(row.X_center + row.Width/2), int(row.Y_center + row.Height/2)), color=(255,0,0), thickness=2)
							if row.Sex == 'male':
								cv2.rectangle(frame, (int(row.X_center - row.Width/2), int(row.Y_center - row.Height/2)), (int(row.X_center + row.Width/2), int(row.Y_center + row.Height/2)), color=(0,0,255), thickness=2)
							if row.TrackID in sub_dt.track_id:
								for time,row in out_dt[out_dt.track_id == row.TrackID]:
									cv2.line(img, (row.X_center, row.Y_center), ((row.x1 + row.x2)/2, (row.y1 + row.y2)/2), (122, 122, 122), 2)
		
					outAll.write(frame)
				else:
					break
					print('VideoError: BadFrame')

			outAll.release()
