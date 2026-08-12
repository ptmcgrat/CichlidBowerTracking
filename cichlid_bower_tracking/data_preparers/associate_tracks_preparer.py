import os, pdb, datetime, shutil
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class AssociateTracksPreparer:
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
		self.fileManager.downloadData(self.fileManager.localAllLabeledClustersFile)
		self.fileManager.downloadData(self.fileManager.localVideoCropFile)
		
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			self.fileManager.downloadData(videoObj.localFishPoseFile)

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localAllLabeledClustersFile)
		assert os.path.exists(self.fileManager.localVideoCropFile)
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localFishPoseFile)

	def uploadData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localAllLabeledClustersFile)
		self.fileManager.uploadData(self.fileManager.localAllTracksSummaryFile)

		if delete:
			os.remove(self.fileManager.localAllLabeledClustersFile)
			os.remove(self.fileManager.localAllTracksSummaryFile)
			os.remove(self.fileManager.localVideoCropFile)

			for videoIndex in self.videoIndices:
				videoObj = self.fileManager.returnVideoObject(videoIndex)
				#os.remove(videoObj.localFishDetectionsFile)


	def createLogFile(self):
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		with open(self.fileManager.localSummaryLogfile,'w') as f:
			print('GitBranch: ' + self.fileManager.branch_name, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

			output = subprocess.run(['conda','list'], capture_output = True)
			print(output.stdout.decode('utf-8'), file = f)


	def createAssociations(self):
		c_dt = pd.read_csv(self.fileManager.localAllLabeledClustersFile, index_col = 0, parse_dates=['TimeStamp'])
		c_dt = c_dt[['ProjectID','VideoID','ClipName','t','X','Y','N','InFrame','ClipCreated','TimeStamp','Prediction','Probability']]
		c_dt['TrackID'] = 0
		c_dt['MinDistance'] = 0
		with open(self.fileManager.localVideoCropFile) as f:
			for line in f:
				video_crop_points = eval(line.rstrip())
		polygon = Polygon(video_crop_points)

		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			dt = pd.read_csv(videoObj.localFishPoseFile)
			dt['ProjectID'] = self.fileManager.projectID
			dt['VideoID'] = videoObj.baseName
			dt['InFrame'] = dt.apply(lambda row: polygon.contains(Point(row['X_center'],row['Y_center'])), axis = 1)
			dt['TimeStamp'] = dt.apply(lambda row: videoObj.startTime + datetime.timedelta(seconds = row.FrameNum / videoObj.framerate), axis = 1)

			for lid, row in c_dt[(c_dt.InFrame == True) & (c_dt.ClipCreated == 'Yes') & (c_dt.VideoID == videoObj.baseName)].iterrows():
				sub_dt = dt[(dt.TimeStamp > row.TimeStamp - datetime.timedelta(seconds = 0.2)) & (dt.TimeStamp < row.TimeStamp + datetime.timedelta(seconds = 0.2))]
				if len(sub_dt) == 0:
					continue
				distance2 = (((row.X - sub_dt['Y_center']) ** 2 + (row.Y - sub_dt['X_center']) ** 2)).min()
	
				track_index = (((row.X - sub_dt['Y_center']) ** 2 + (row.Y - sub_dt['X_center']) ** 2)).idxmin()
				try:
					c_dt.loc[lid,'TrackID'] = int(dt.loc[track_index]['TrackID'])
					c_dt.loc[lid,'MinDistance'] = int(distance2 ** 0.5)
					
				except:
					pdb.set_trace()		
			
			track_dt = dt.groupby(['ProjectID','VideoID','TrackID']).agg(nFrames=('TrackID','count'), aggSex=('SexID','mean'), aggInFrame = ('InFrame','mean')).reset_index()
			try:
				# c_dt = c_dt.append(new_dt)
				s_track_dt = pd.concat([s_track_dt, track_dt], ignore_index=True)
			except NameError:
				s_track_dt = track_dt
		
		s_track_dt['Sex'] = 'Male'
		s_track_dt.loc[s_track_dt.aggSex <= 0.5,'Sex'] = 'Female'
		s_track_dt['InFrameTrack'] = True
		s_track_dt.loc[s_track_dt.aggInFrame <= 0.5,'InFrameTrack'] = False
		s_track_dt.to_csv(self.fileManager.localAllTracksSummaryFile)
		c_dt = pd.merge(c_dt, s_track_dt[['ProjectID','VideoID','TrackID','Sex','InFrameTrack']], on = ['ProjectID','VideoID','TrackID'], how = 'left')
		c_dt.loc[c_dt.TrackID==0, 'Sex'] = 'Unknown'
		c_dt.loc[c_dt.TrackID==0, 'InFrameTrack'] = False
		c_dt = c_dt[['ProjectID','VideoID','ClipName','t','X','Y','N','InFrame','ClipCreated','TimeStamp','Prediction','Probability','TrackID','MinDistance','Sex','InFrameTrack']]
		
		c_dt.to_csv(self.fileManager.localAllLabeledClustersFile)

