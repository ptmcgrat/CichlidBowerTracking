import os, pdb
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

class ClusterTrackAssociationPreparer():

	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.validateInputData()

	def validateInputData(self):
		
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localOldVideoCropFile)
		assert os.path.exists(self.fileManager.localAllLabeledClustersFile)
		return
		for videoIndex in range(len(self.fileManager.lp.movies)):
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localFishTracksFile)
			assert os.path.exists(videoObj.localFishDetectionsFile)

	def summarizeTracks(self):

		# Loop through videos and combine into a single file
		for videoIndex in range(len(self.fileManager.lp.movies)):
			videoObj = self.fileManager.returnVideoObject(videoIndex)

			#Read in individual video detection and tracks files
			video_dt_t = pd.read_csv(videoObj.localFishTracksFile)
			video_dt_d = pd.read_csv(videoObj.localFishDetectionsFile)

			# Combine them into a single master pandas DataFrame
			try:
				dt_t = c_dt_t.append(video_dt_t)
				dt_d = c_dt_d.append(video_dt_d)
			except NameError:
				dt_t = video_dt_t
				dt_d = video_dt_d

		# Save raw detections file
		dt_d.to_csv(self.fileManager.localAllFishDetectionsFile)

		# Use video_crop to determine if fish is inside or outside the frame
		video_crop = np.load(self.fileManager.localOldVideoCropFile)
		poly = Polygon(video_crop)
		dt_t['InBounds'] = [poly.contains(Point(x, y)) for x,y in zip(dt_t.xc, dt_t.yc)]

		# Determine track lengths (useful for identifing longest tracks for manual annotation)
		track_lengths = dt_t.groupby(['track_id','base_name']).count()['p_value'].rename('track_length').reset_index()
		dt_t = pd.merge(dt_t, track_lengths, left_on = ['track_id','base_name'], right_on = ['track_id','base_name'])
		#dt_t['binned_track_length'] = dt_t.track_length.apply(bin_tracklength)

		dt_t.to_csv(self.fileManager.localAllFishTracksFile)

		pdb.set_trace()
		t_dt = t_dt.groupby(['track_id', 'track_length', 'base_name']).mean()[['class', 'p_value','InBounds']].rename({'class':'SexCall'}, axis = 1).reset_index().sort_values(['base_name','track_id'])
		t_dt.to_csv(self.fileManaer.localAllTracksSummaryFile, index = False)

	def associateClustersWithTracks(self):
		c_dt = pd.read_csv(self.fm.localAllLabeledClustersFile)
		pdb.set_trace()

	def createMaleFemaleAnnotationVideos(self):
		s_dt = pd.read_csv(self.fileManager.localAllTracksSummaryFile)
		t_dt = pd.read_csv(self.fileManager.localAllFishTracksFile)
		pdb.set_trace()
		# Group data together to single track
