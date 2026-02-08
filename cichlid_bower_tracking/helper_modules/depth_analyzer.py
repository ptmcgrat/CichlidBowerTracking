import math
import pandas as pd
import numpy as np
import datetime, sys, pdb
from skimage import morphology
from types import SimpleNamespace
from PIL import Image,ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class DepthAnalyzer:
	# Contains code process depth data for figure creation

	def __init__(self, fileManager):
		self.fileManager = fileManager
		self.lp = self.fileManager.lp
		self.first_good_index = self.lp.frames
		# pdb.set_trace()
		self._loadData()
		
		self.goodPixels = np.count_nonzero(~np.isnan(self.depth_data[0,:]))

	def _loadData(self):
		# Loads depth tray information and smoothedDepthData from files that have already been downloaded

		# try:
		#     self.depth_data
		# except AttributeError:
		#     self.depth_data = np.load(self.fileManager.localSmoothDepthFile)
		try:
			self.depth_data
		except AttributeError:
			self.depth_data = np.load(self.fileManager.localSmoothDepthFile)
		self.num_frames, self.height, self.width = self.depth_data.shape

	def t_to_index(self, t):
		try:
			index = max([False if x.time <= t else True for x in self.lp.frames].index(True) - 1, 0)
		except ValueError:
			if t > self.lp.frames[-1].time:
				index = len(self.lp.frames) - 1
			else:
				index = 0
		print(index)
		return index

	def clip_data(self, t0, t1):
		# clips the data and log parser to a particular time range. Useful for reducing the size of
		# the DepthAnalyzer when generating multiple DepthAnalyzer objects
		self._checkTimes(t0, t1)
		i0, i1 = (self.t_to_index(t) for t in [t0, t1])
		self.depth_data = self.depth_data[i0:i1 + 1]
		self.lp.frames = self.lp.frames[i0:i1+1]

	def returnBowerLocations(self, t0, t1, thresh = None, cropped=True, force_window=False):
		# Returns 2D numpy array using thresholding and minimum size data to identify bowers
		# Pits = -1, Castle = 1, No bower = 0

		# Check times are good
		self._checkTimes(t0, t1)

		# Identify total height change and time change
		totalHeightChange = self.returnHeightChange(t0, t1, masked=False, cropped=cropped, force_window=force_window)
		timeChange = t1 - t0

		# Determine threshold and minimum size of bower to use based upon timeChange
		if t0 > t1:
			totalThreshold = self.fileManager.totalDepthThreshold
			minPixels = self.fileManager.totalMinPixels
		elif timeChange.total_seconds() < 7300:  # 2 hours or less
			totalThreshold = self.fileManager.hourlyDepthThreshold
			minPixels = self.fileManager.hourlyMinPixels
		elif timeChange.total_seconds() < 129600:  # 2 hours to 1.5 days
			totalThreshold = self.fileManager.dailyDepthThreshold
			minPixels = self.fileManager.dailyMinPixels
		else:  # 1.5 days or more
			totalThreshold = self.fileManager.totalDepthThreshold
			minPixels = self.fileManager.totalMinPixels
		if thresh is not None:
			totalThreshold = thresh

		tCastle = np.where(totalHeightChange >= totalThreshold, True, False)
		tCastle = morphology.remove_small_objects(tCastle, max_size = minPixels).astype(int)

		tPit = np.where(totalHeightChange <= -1 * totalThreshold, True, False)
		tPit = morphology.remove_small_objects(tPit, max_size = minPixels).astype(int)

		bowers = (tCastle - tPit).astype('float')
		bowers[np.isnan(totalHeightChange)] = np.nan
		return bowers

	def returnHeight(self, t, cropped=False):
		# return the frame from the smoothedDepthData numpy closest to time t. If cropped is True, crop the frame
		# to include only the area defined by tray_r

		# Check times are good
		self._checkTimes(t)

		# Find closest frames to desired times
		try:
			first_index = max([False if x.time <= t else True for x in self.lp.frames].index(True) - 1,
							  0)  # This ensures that we get overnight changes when kinect wasn't running
		except ValueError:
			if t > self.lp.frames[-1].time:
				first_index = -1
			else:
				first_index = 0

		change = self.depth_data[first_index]

		return change

	def returnHeightChange(self, t0, t1, masked=False, cropped=False, force_window=False):
		# return the height change, based on the smoothedDepthData numpy, from the frame closest to t0 to the frame
		# closest to t1. If cropped is True, crop the frame to include only the area defined by tray_r. If masked is
		# True, set the pixel value in all non-bower regions (see returnBowerLocations) to 0

		# Check times are good
		self._checkTimes(t0, t1)

		# Find closest frames to desired times
		try:
			first_index = max([False if x.time <= t0 else True for x in self.lp.frames].index(True) - 1,
							  0)  # This ensures that we get overnight changes when kinect wasn't running
		except ValueError:
			if t0 > self.lp.frames[-1].time:
				first_index = -1
			else:
				first_index = 0
		if force_window:
			while self.lp.frames[first_index].time < t0:
				first_index += 1

		try:
			last_index = max([False if x.time <= t1 else True for x in self.lp.frames].index(True) - 1, 0)
		except ValueError:
			last_index = len(self.lp.frames) - 1

		#print(str(first_index) + ': ' + str(last_index))
		change = self.depth_data[first_index] - self.depth_data[last_index]

		if masked:
			change[self.returnBowerLocations(t0, t1, cropped=cropped) == 0] = 0



		return change

	def returnVolumeSummary(self, t0, t1, thresh = None):
		# calculate various summary statistics for the depth change from t0 to t1

		# Check times are good
		self._checkTimes(t0, t1)

		pixelLength = self.fileManager.pixelLength
		bowerIndex_pixels = int(self.goodPixels * self.fileManager.bowerIndexFraction)

		bowerLocations = self.returnBowerLocations(t0, t1, thresh)


		heightChange = self.returnHeightChange(t0, t1)
		heightChangeAbs = heightChange.copy()
		heightChangeAbs = np.abs(heightChangeAbs)

		outData = SimpleNamespace()
		# Get data
		outData.projectID = self.lp.projectID
		outData.depthAbsoluteVolume = np.nansum(heightChangeAbs) * pixelLength ** 2
		outData.depthSummedVolume = np.nansum(heightChange) * pixelLength ** 2
		outData.depthCastleArea = np.count_nonzero(bowerLocations == 1) * pixelLength ** 2
		outData.depthPitArea = np.count_nonzero(bowerLocations == -1) * pixelLength ** 2
		outData.depthCastleVolume = np.nansum(heightChange[bowerLocations == 1]) * pixelLength ** 2
		outData.depthPitVolume = np.nansum(heightChange[bowerLocations == -1]) * -1 * pixelLength ** 2
		outData.depthBowerVolume = outData.depthCastleVolume + outData.depthPitVolume

		flattenedData = heightChangeAbs.flatten()
		sortedData = np.sort(flattenedData[~np.isnan(flattenedData)])
		try:
			threshold = sortedData[-1 * bowerIndex_pixels]
		except IndexError:
			print('Warning: threshold error')
			threshold = 0
		outData.thresholdCastleVolume = np.nansum(heightChangeAbs[(bowerLocations == 1) & (heightChangeAbs > threshold)])
		outData.thresholdPitVolume = np.nansum(heightChangeAbs[(bowerLocations == -1) & (heightChangeAbs > threshold)])

		outData.depthBowerIndex = (outData.thresholdCastleVolume - outData.thresholdPitVolume) / (outData.thresholdCastleVolume + outData.thresholdPitVolume)

		return outData

	def _checkTimes(self, t0, t1=None):
		# validate the given times
		if t1 is None:
			if type(t0) != datetime.datetime:
				try:
					t0 = t0.to_pydatetime()
				except AttributeError:
					raise Exception('Timepoints to must be datetime.datetime objects')
			return
		# Make sure times are appropriate datetime objects
		if type(t0) != datetime.datetime or type(t1) != datetime.datetime:
			try:
				t0 = t0.to_pydatetime()
				t1 = t1.to_pydatetime()
			except AttributeError:
				raise Exception('Timepoints to must be datetime.datetime objects')
		#if t0 > t1:
		#    print('Warning: Second timepoint ' + str(t1) + ' is earlier than first timepoint ' + str(t0),
		#          file=sys.stderr)

class ClusterAnalyzer:
	# Contains code process cluster data for figure creation
	def __init__(self, fileManager):
		self.fileManager = fileManager
		self.bids = ['c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x']
		self.bid_labels = {'c':'bower scoop', 'p': 'bower spit', 'b': 'bower multiple',
						   'f': 'feed scoop', 't': 'feed spit', 'm': 'feed multiple',
						   's': 'spawn', 'd': 'drop sand', 'o': 'fish other', 'x': 'no fish other'}
		self.bid_colors = {'c':(0,0,255), 'p': (0,0,255), 'b': (0,0,255),
						   'f': (255, 165,0), 't': (255, 165,0), 'm': (255, 165,0),
						   's': (255, 192, 203), 'd': (0,255,0), 'o': (0,255,0), 'x': (187, 194, 194)}

		self.lp = self.fileManager.lp
		self._loadData()

	def _loadData(self):
		# load the required data

		self.transM = np.load(self.fileManager.localTransMFile)
		self.clusterData = pd.read_csv(self.fileManager.localAllLabeledClustersFile, index_col='TimeStamp',
									   parse_dates=True)
		self._appendDepthCoordinates()
		
		with open(self.fileManager.localVideoCropFile) as f:
			for line in f:
				video_crop_points = eval(line.rstrip())
		polygon = Polygon(video_crop_points)
		
		buffered_polygon = polygon.buffer(20, join_style=2)
		self.clusterData['InFrame'] = self.clusterData.apply(lambda row: buffered_polygon.contains(Point(row['X'],row['Y'])), axis = 1)

		self.clusterData.to_csv(self.fileManager.localAllLabeledClustersFile)

	def _appendDepthCoordinates(self):
		# adds columns containing X and Y in depth coordinates to all cluster csv
		self.clusterData['X_depth'] = self.clusterData.apply(
			lambda row: (self.transM[0][0] * row.Y + self.transM[0][1] * row.X + self.transM[0][2]) / (
					self.transM[2][0] * row.Y + self.transM[2][1] * row.X + self.transM[2][2]), axis=1)
		self.clusterData['Y_depth'] = self.clusterData.apply(
			lambda row: (self.transM[1][0] * row.Y + self.transM[1][1] * row.X + self.transM[1][2]) / (
					self.transM[2][0] * row.Y + self.transM[2][1] * row.X + self.transM[2][2]), axis=1)
		#scaling_factor = math.sqrt(np.linalg.det(self.transM))
		#self.clusterData['approx_radius'] = self.clusterData.apply(
		#	lambda row: (np.mean(row.X_span + row.Y_span) * scaling_factor)/2, axis=1)
		# self.clusterData.round({'X_Depth': 0, 'Y_Depth': 0})


	def returnDepthCoordinates(self, t0=None, t1=None, bid=None, in_frame=True, created = True, confidence = None, earlyEvents = False, lateEvents = False, N = None, sex = None):
		# utility function to return two numpy arrays that match the provided criteria
		# t0: return only rows with timestamps after t0
		# t1: return only rows with timestamps before t1
		# bid: single letter character matching the behavioral id
		# cropped: If True, events that occur within the area defined by the video crop
		dt = self.clusterData
		dt = dt.sort_index()
		if sex is not None:
			dt.loc[dt.MinDistance > 100,'Sex'] = 'Unknown'
			dt = dt[dt.Sex==sex]
		if confidence is not None:
			dt = dt[dt.Probability>confidence]
		if N is not None:
			dt = dt[dt.N > N]
		if created:
			dt = dt[dt.ClipCreated=='Yes']
		elif created is None:
			dt = dt
		else:
			dt = dt[dt.ClipCreated=='No']
		if in_frame:
			dt = dt[dt.InFrame == True]
		elif in_frame is None:
			dt = dt
		else:
			dt = dt[dt.InFrame == False]
		if t0 is not None:
			self._checkTimes(t0, t1)
			try:
				if earlyEvents:
					t0 = t0.replace(hour = 8, minute=0, second=0, microsecond=0)
					t1 = t1.replace(hour = 14, minute=0, second=0, microsecond=0)
				elif lateEvents:
					t0 = t0.replace(hour = 14, minute=0, second=0, microsecond=0)
					t1 = t1.replace(hour = 20, minute=0, second=0, microsecond=0)
				dt = dt[t0:t1]
			except KeyError:
				pdb.set_trace()
		if bid is not None:
			if isinstance(bid,list):
				dt = dt[dt.Prediction.isin(bid)]
			else:
				dt = dt[dt.Prediction == bid]
		return [np.array(dt.X_depth), np.array(dt.Y_depth)]

	def returnCategoryCounts(self, t0=None, t1=None, cropped=True):
		# utility function to returns the number of counts for each category along with two summary categories
		# t0: return only rows with timestamps after t0
		# t1: return only rows with timestamps before t1
		# cropped: If True, events that occur within the area defined by the video crop
		output = {}
		combined_output = {}
		dt = self.clusterData
		dt = dt.dropna(subset=['Prediction']).sort_index()

		if t0 is not None:
			self._checkTimes(t0, t1)
			dt = dt[t0:t1]
		if cropped:
			dt = dt[dt.InFrame == True]
		for bid in self.bids:
			output[bid] = len(dt[dt.Prediction == bid])
		combined_output['Building'] = output['c'] + output['p'] + output['b']
		combined_output['Feeding'] = output['f'] + output['t'] + output['m']
		combined_output['Spawning'] = output['s']
		combined_output['Other'] = output['d'] + output['o'] + output['x']
		return output, combined_output

	def addClusterLabels(self, current_time, videoObj, delta_t = 1, delta_xy = 30):
		dt = self.clusterData
		min_time = current_time - datetime.timedelta(seconds=delta_t)
		max_time = current_time + datetime.timedelta(seconds=delta_t)
		dt = dt[(dt.index > min_time) & (dt.index < max_time)][['X','Y','ClipCreated','Prediction','InFrame','TrackID']]
		out = pd.DataFrame(columns = ['x1','x2','y1','y2','color','label', 'track_id'])
		for time,row in dt.iterrows():
			if row.ClipCreated == 'No':
				color = (0,0,0)
				label = 'NoClip'
			elif not row.InFrame:
				color = (255,0,0)
				label = 'Cropped'
			else:
				try:
					color = self.bid_colors[row.Prediction]
					label = self.bid_labels[row.Prediction]
				except KeyError:
					print(row)
					color = (255,255,255)
					label = 'UnknownError'
		
			out.loc[len(out)] = [max(0,row.Y-delta_xy), min(videoObj.width,row.Y+delta_xy),
								 max(0,row.X-delta_xy), min(videoObj.height,row.X+delta_xy),
								 color, label, row.TrackID]
		return out

	def sliceDataframe(self, t0=None, t1=None, bid=None, cropped=True):
		# utility function to access specific slices of the Dataframe based on the AllClusterData csv.
		#
		# t0: return only rows with timestamps after t0
		# t1: return only rows with timestamps before t1
		# bid: string or list of strings: return only rows for which the behavioral id matches the given string(s)
		# columns: list of strings: return only the columns of data matching the given keys
		# input_frame: pd.DataFrame: dataframe to slice from, instead of the default full dataframe.
		#              Allows for iterative slicing
		# cropped: If True, return only rows corresponding to events that occur within the area defined by tray_r

		df_slice = self.clusterData if input_frame is None else input_frame
		df_slice = df_slice.dropna(subset=['Prediction']).sort_index()
		if t0 is not None:
			self._checkTimes(t0, t1)
			df_slice = df_slice[t0:t1]
		if bid is not None:
			df_slice = df_slice[df_slice.Prediction.isin(bid if type(bid) is list else [bid])]
		if cropped:
			df_slice = df_slice[(df_slice.X_depth > self.tray_r[0]) & (df_slice.X_depth < self.tray_r[2]) &
								(df_slice.Y_depth > self.tray_r[1]) & (df_slice.Y_depth < self.tray_r[3])]
			df_slice.X_depth = df_slice.X_depth - self.tray_r[0]
			df_slice.Y_depth = df_slice.Y_depth - self.tray_r[1]
		if columns is not None:
			df_slice = df_slice[columns]
		return df_slice

	def returnClusterCounts(self, t0, t1, bid='all', cropped=True):
		# return the number of behavioral events for a given behavior id (bid), or all bids, between t0 and t1
		#
		# t0: beginning of desired time frame
		# t1: end of desired time frame
		# bid: string or 'all': bid that will be counted. if a single bid is given, return counts (as an int) for that
		#      bid only. If 'all' (default behavior) return a dict of counts for all bids, keyed by bid.
		# cropped: If True, count only events occuring within the area defined by tray_r
		self._checkTimes(t0, t1)
		if bid == 'all':
			df_slice = self.sliceDataframe(t0=t0, t1=t1, cropped=cropped)
			row = df_slice.Prediction.value_counts().to_dict
			return row
		else:
			df_slice = self.sliceDataframe(t0=t0, t1=t1, bid=bid, cropped=cropped)
			cell = df_slice.Prediction.count()
			return cell

	def returnClusterKDE(self, t0, t1, bid, cropped=True, bandwidth=None):
		# Geneate a kernel density estimate corresponding to the number events per cm^2 over a given timeframe for
		# a particular behavior id (bid)
		#
		# t0: beginning of time frame
		# t1: end of time frame
		# bid: string: generate a kde for this bid
		# cropped: if True, only include events within the area defined by tray_r
		# bandwidth: Can be used to manually set the kde bandwith (see sklearn.neighbors.KernelDensity). By default,
		#            use a bandwidth based on the average approximate event radius (see _appendDepthCoordinates)
		if bandwidth is None:
			bandwidth = self.sliceDataframe(t0, t1, bid, 'approx_radius').mean()/2
		df_slice = self.sliceDataframe(t0=t0, t1=t1, bid=bid, cropped=cropped, columns=['X_depth', 'Y_depth'])
		n_events = len(df_slice.index)
		x_bins = int(self.tray_r[2] - self.tray_r[0])
		y_bins = int(self.tray_r[3] - self.tray_r[1])
		xx, yy = np.mgrid[0:x_bins, 0:y_bins]
		if n_events == 0:
			z = np.zeros_like(xx)
		else:
			xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
			xy_train = df_slice.to_numpy()
			kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(xy_train)
			z = np.exp(kde.score_samples(xy_sample)).reshape(xx.shape)
			z = (z * n_events) / (z.sum() * (self.fileManager.pixelLength ** 2))
		return z

	def returnBowerLocations(self, t0, t1, cropped=True, bandwidth=None):
		# Returns 2D numpy array using thresholding and minimum size data to identify bowers based on KDEs of spit and
		# scoop densities. Pits = -1, Castle = 1, No bower = 0
		#
		# t0: beginning of time frame
		# t1: end of time frame
		# cropped: if True, include in bower calculation only events within the area defined by tray_r
		# bandwith: see returnClusterKDE

		self._checkTimes(t0, t1)
		timeChange = t1 - t0

		if timeChange.total_seconds() < 7300:  # 2 hours or less
			totalThreshold = self.fileManager.hourlyClusterThreshold
			minPixels = self.fileManager.hourlyMinPixels
		elif timeChange.total_seconds() < 129600:  # 2 hours to 1.5 days
			totalThreshold = self.fileManager.dailyClusterThreshold
			minPixels = self.fileManager.dailyMinPixels
		else:  # 1.5 days or more
			totalThreshold = self.fileManager.totalClusterThreshold
			minPixels = self.fileManager.totalMinPixels

		z_scoop = self.returnClusterKDE(t0, t1, 'c', cropped=cropped, bandwidth=bandwidth)
		z_spit = self.returnClusterKDE(t0, t1, 'p', cropped=cropped, bandwidth=bandwidth)

		scoop_binary = np.where(z_spit - z_scoop <= -1 * totalThreshold, True, False)
		scoop_binary = morphology.remove_small_objects(scoop_binary, max_size = minPixels).astype(int)

		spit_binary = np.where(z_spit - z_scoop >= totalThreshold, True, False)
		spit_binary = morphology.remove_small_objects(spit_binary, max_size = inPixels).astype(int)

		bowers = spit_binary - scoop_binary
		return bowers

	def returnClusterSummary(self, t0, t1):
		# calculate various summary statistics for the scoop and spit KDEs from t0 to t1
		self._checkTimes(t0, t1)
		pixelLength = self.fileManager.pixelLength
		bowerIndex_pixels = int(self.goodPixels * self.fileManager.bowerIndexFraction)
		bowerLocations = self.returnBowerLocations(t0, t1)
		clusterKde = self.returnClusterKDE(t0, t1, 'p') - self.returnClusterKDE(t0, t1, 'c')
		clusterKdeAbs = clusterKde.copy()
		clusterKdeAbs = np.abs(clusterKdeAbs)

		outData = SimpleNamespace()
		# Get data
		outData.projectID = self.lp.projectID
		outData.kdeAbsoluteVolume = np.nansum(clusterKdeAbs) * pixelLength ** 2
		outData.kdeSummedVolume = np.nansum(clusterKde) * pixelLength ** 2
		outData.kdeCastleArea = np.count_nonzero(bowerLocations == 1) * pixelLength ** 2
		outData.kdePitArea = np.count_nonzero(bowerLocations == -1) * pixelLength ** 2
		outData.kdeCastleVolume = np.nansum(clusterKde[bowerLocations == 1]) * pixelLength ** 2
		outData.kdePitVolume = np.nansum(clusterKde[bowerLocations == -1]) * -1 * pixelLength ** 2
		outData.kdeBowerVolume = outData.kdeCastleVolume + outData.kdePitVolume

		flattenedData = clusterKdeAbs.flatten()
		sortedData = np.sort(flattenedData[~np.isnan(flattenedData)])
		try:
			threshold = sortedData[-1 * bowerIndex_pixels]
		except IndexError:
			threshold = 0
		thresholdCastleKdeVolume = np.nansum(clusterKdeAbs[(bowerLocations == 1) & (clusterKdeAbs > threshold)])
		thresholdPitKdeVolume = np.nansum(clusterKdeAbs[(bowerLocations == -1) & (clusterKdeAbs > threshold)])

		outData.kdeBowerIndex = (thresholdCastleKdeVolume - thresholdPitKdeVolume) / (thresholdCastleKdeVolume + thresholdPitKdeVolume)

		return outData

	def _checkTimes(self, t0, t1=None):
		# validate the given times
		if t1 is None:
			if type(t0) != datetime.datetime:
				try:
					t0 = t0.to_pydatetime()
				except AttributeError:
					raise Exception('Timepoints to must be datetime.datetime objects')
			return
		# Make sure times are appropriate datetime objects
		if type(t0) != datetime.datetime or type(t1) != datetime.datetime:
			try:
				t0 = t0.to_pydatetime()
				t1 = t1.to_pydatetime()
			except AttributeError:
				raise Exception('Timepoints to must be datetime.datetime objects')
		#if t0 > t1:
			#print('Warning: Second timepoint ' + str(t1) + ' is earlier than first timepoint ' + str(t0),
			#      file=sys.stderr)
