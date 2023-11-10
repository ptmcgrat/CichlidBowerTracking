import scipy.signal
import skvideo.io
import numpy as np
import pdb, os, sys, datetime, warnings, copy, subprocess
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image,ImageDraw
from cichlid_bower_tracking.helper_modules.depth_analyzer import DepthAnalyzer as DA
from collections import OrderedDict
from matplotlib import (cm, colors, gridspec, ticker)
import seaborn as sns
import pandas as pd 

warnings.filterwarnings('ignore')



class DepthPreparer:
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, workers = None):
		
		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.device = self.fileManager.lp.device
		self.createLogFile()

	def downloadProjectData(self):
		self.createDirectory(self.localMasterDir)
		self.createDirectory(self.localAnalysisDir)
		self.createDirectory(self.localTroubleshootingDir)
		self.createDirectory(self.localLogfileDir)
		self.createDirectory(self.localSummaryDir)

		self.downloadData(self.localLogfile)
		self.downloadData(self.localFrameDir, tarred = True)
		self.downloadData(self.localDepthCropFile)


	def validateInputData(self):
		assert os.path.exists(self.fileManager.localLogfile)
		self.lp = self.fileManager.lp
		bad_frames = 0
		for frame in self.lp.frames:
			if not os.path.exists(self.fileManager.localProjectDir + frame.npy_file):
				bad_frames += 1
			if not os.path.exists(self.fileManager.localProjectDir + frame.pic_file):
				bad_frames += 1
		#print(bad_frames)
		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localDepthCropFile)

	def createLogFile(self):
		with open(self.fileManager.localDepthLogfile,'w') as f:
			print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
			print('NumpyVersion: ' + np.__version__, file = f)
			print('Scikit-VideoVersion: ' + skvideo.__version__, file = f)
			print('ScipyVersion: ' + scipy.__version__, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

	def createSmoothedArray(self, goodDataCutoff = 0.8, minimumGoodData = 0.95, tunits = 71, order = 4, max_depth = 12, max_height = 12):
		
		for i in range(12): #Loop through each day preceding euthanization
			frames = [x for x in self.lp.frames if x.lof == True]


		# Create arrays to store raw depth data and data in the daytime
		rawDepthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
		daytimeData = np.empty(shape = (len(frames), self.lp.height, self.lp.width))

		# Read in each frame and store it. Also keep track of the indeces that are in the daytime
		day_idx = 0
		day_start_stop = OrderedDict() # Dictionary to hold first and last frame indeces for good and bad data for each day
		
		ad_idx = 0
		for i, frame in enumerate(self.lp.frames):
			try:
				data = np.load(self.fileManager.localProjectDir + frame.npy_file)
				day = frame.time.day

			except FileNotFoundError:
				print('Bad frame: ' + str(i) + ', ' + frame.npy_file)
				rawDepthData[i] = rawDepthData[i-1]
			else:
				rawDepthData[i] = data

			if frame.lof:
				# Daytime frame
				daytimeData[day_idx] = rawDepthData[i]
				day_idx += 1

				# Store first and last frame for each day
				if day not in day_start_stop:
					day_start_stop[day] = [i,i]
				else:
					day_start_stop[day][1] = i

		# Save raw data file
		np.save(self.fileManager.localRawDepthFile, rawDepthData)

		# Interpolate missing data
		# Make copy of raw data
		interpDepthData = rawDepthData.copy()

		# Loop through each day and interpolate missing data
		for day,(start_index,stop_index) in day_start_stop.items():
			dailyData = interpDepthData[start_index:stop_index] # Create view of numpy array just creating a single day during the daytime
			goodDataAll = np.count_nonzero(~np.isnan(dailyData), axis = 0)/dailyData.shape[0] # Calculate the fraction of good data points per pixel

			# Process each pixel
			for i in range(dailyData.shape[1]):
				for j in range(dailyData.shape[2]):
					if goodDataAll[i,j] > goodDataCutoff: # If enough data is present in the pixel then interpolate
				
						x_interp, = np.where(np.isnan(dailyData[:,i,j])) # Indices with missing data
						x_good, = np.where(~np.isnan(dailyData[:,i,j])) # Indices with good data

						if len(x_interp) != 0: # Only interpolate if there is missing data
							interp_data = np.interp(x_interp, x_good, dailyData[x_good, i, j])
							dailyData[x_interp, i, j] = interp_data
		
		# Save interpolated data
		np.save(self.fileManager.localInterpDepthFile, interpDepthData)

		# Smooth and filter out bad data 
		smoothDepthData = interpDepthData.copy()
		
		# Read in manual crop and mask out data outside of crop
		with open(self.fileManager.localDepthCropFile) as f:
			for line in f:
				depth_crop_points = eval(line.rstrip())

		img = Image.new('L', (self.lp.width, self.lp.height), 0)
		ImageDraw.Draw(img).polygon(depth_crop_points, outline=1, fill=1)
		manual_crop_mask = np.array(img)
		smoothDepthData[:,manual_crop_mask == 0] = np.nan


		# Mask out data with too many nans
		non_nans = np.count_nonzero(~np.isnan(daytimeData), axis = 0)
		smoothDepthData[:,non_nans < minimumGoodData*daytimeData.shape[0]] = np.nan

		# Filter out data with bad standard deviations
		stds = np.nanstd(daytimeData, axis = 0)
		smoothDepthData[:,stds > 1.5] = np.nan # Filter out data with std > 1.5 cm

		# Filter out data that is too close or too far from the sensor
		average_depth = np.nanmean(daytimeData, axis = 0)
		median_height = np.nanmedian(average_depth)
		smoothDepthData[:,(average_depth > median_height + max_depth) | (average_depth < median_height - max_height)] = np.nan # Filter out data 4cm lower and 8cm higher than tray


		# Nighttime data is bad. Set it to average of data before and after.
		boundaries = np.array(list(day_start_stop.values())).flatten()
		for i, frame in enumerate(self.lp.frames):
			if not frame.lof:
				closest_idx = min(range(len(boundaries)), key=lambda j: abs(boundaries[j]-i))
				closest_value = boundaries[closest_idx]
				if closest_idx == 0:
					smoothDepthData[i] = smoothDepthData[boundaries[0]]
				elif closest_idx == len(boundaries) - 1:
					smoothDepthData[i] = smoothDepthData[boundaries[-1]]
				else:
					if i - closest_value > 0:
						smoothDepthData[i] = np.nanmean((smoothDepthData[boundaries[closest_idx]],smoothDepthData[boundaries[closest_idx+1]]), axis = 0)
					else:
						smoothDepthData[i] = np.nanmean((smoothDepthData[boundaries[closest_idx]],smoothDepthData[boundaries[closest_idx-1]]), axis = 0)

		# Smooth data with savgol_filter
		smoothDepthData = scipy.signal.savgol_filter(smoothDepthData, tunits, order, axis = 0, mode = 'mirror')
		np.save(self.fileManager.localSmoothDepthFile, smoothDepthData)

	def createDepthFigures(self, hourlyDelta=2):

		print('Creating Depth Figure')
		# Create all figures based on depth data. Adjust hourlyDelta to influence the resolution of the
		# HourlyDepthSummary.pdf figure

		# Check that the DepthAnalzer object has been created, indicating that the required files are present.
		# Otherwise, skip creation of Depth Figures
		self.da_obj = DA(self.fileManager)

		# figures based on the depth data

		# Create summary figure of daily values
		figDaily = plt.figure(num=1, figsize=(11, 8.5))
		figDaily.suptitle(self.lp.projectID + ' Daily Depth Summary')
		gridDaily = gridspec.GridSpec(3, 1)

		# Create summary figure of hourly values
		figHourly = plt.figure(num=2, figsize=(11, 8.5))
		figHourly.suptitle(self.lp.projectID + ' Hourly Depth Summary')

		start_day = self.lp.frames[0].time.replace(hour=0, minute=0, second=0, microsecond=0)
		totalChangeData = vars(self.da_obj.returnVolumeSummary(self.lp.frames[0].time, self.lp.frames[-1].time))

		# Show picture of final depth
		topGrid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gridDaily[0])
		topAx1 = figDaily.add_subplot(topGrid[0])
		topAx1_ax = topAx1.imshow(self.da_obj.returnHeight(self.lp.frames[-1].time, cropped=True), vmin=50, vmax=70)
		topAx1.set_title('Final Depth (cm)')
		topAx1.tick_params(colors=[0, 0, 0, 0])
		plt.colorbar(topAx1_ax, ax=topAx1)

		# Show picture of total depth change
		topAx2 = figDaily.add_subplot(topGrid[1])
		topAx2_ax = topAx2.imshow(self.da_obj.returnHeightChange(
			self.lp.frames[0].time, self.lp.frames[-1].time, cropped=True), vmin=-5, vmax=5)
		topAx2.set_title('Total Depth Change (cm)')
		topAx2.tick_params(colors=[0, 0, 0, 0])
		plt.colorbar(topAx2_ax, ax=topAx2)

		# Show picture of pit and castle mask
		topAx3 = figDaily.add_subplot(topGrid[2])
		topAx3_ax = topAx3.imshow(self.da_obj.returnHeightChange(self.lp.frames[0].time, self.lp.frames[-1].time, cropped = True, masked = True), vmin = -5, vmax = 5)
		topAx3.set_title('Masked Depth Change (cm)')
		topAx3.tick_params(colors=[0, 0, 0, 0])
		plt.colorbar(topAx3_ax, ax=topAx3)

		# Create figures and get data for daily Changes
		dailyChangeData = []
		w_ratios = ([1.0] * self.lp.numDays) + [0.25]
		midGrid = gridspec.GridSpecFromSubplotSpec(3, self.lp.numDays + 1, subplot_spec=gridDaily[1], width_ratios=w_ratios)
		v = 4
		for i in range(self.lp.numDays):
			start = start_day + datetime.timedelta(hours=24 * i)
			stop = start_day + datetime.timedelta(hours=24 * (i + 1))
			dailyChangeData.append(vars(self.da_obj.returnVolumeSummary(start, stop)))
			dailyChangeData[i]['Day'] = i + 1
			dailyChangeData[i]['Midpoint'] = i + 1 + .5
			dailyChangeData[i]['StartTime'] = str(start)

			current_axs = [figDaily.add_subplot(midGrid[n, i]) for n in [0, 1, 2]]
			current_axs[0].imshow(self.da_obj.returnHeightChange(start_day, stop, cropped=True), vmin=-v, vmax=v)
			current_axs[0].set_title('Day %i' % (i + 1))
			current_axs[1].imshow(self.da_obj.returnHeightChange(start, stop, cropped=True), vmin=-v, vmax=v)
			current_axs[2].imshow(self.da_obj.returnHeightChange(start, stop, masked=True, cropped=True), vmin=-v, vmax=v)
			[ax.tick_params(colors=[0, 0, 0, 0]) for ax in current_axs]
			[ax.set_adjustable('box') for ax in current_axs]
		cax = figDaily.add_subplot(midGrid[:, -1])
		plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)

		figHourly = plt.figure(figsize=(11, 8.5))
		gridHourly = plt.GridSpec(self.lp.numDays, int(11 / hourlyDelta) + 2, wspace=0.05, hspace=0.05)
		bounding_ax = figHourly.add_subplot(gridHourly[:, :])
		bounding_ax.xaxis.set_visible(False)
		bounding_ax.set_ylabel('Day')
		bounding_ax.set_ylim(self.lp.numDays + 0.5, 0.5)
		bounding_ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
		bounding_ax.set_yticklabels(range(self.lp.numDays + 1))
		sns.despine(ax=bounding_ax, left=True, bottom=True)

		hourlyChangeData = []
		v = 1
		for i in range(0, self.lp.numDays):
			current_j = 0
			for j in range(int(24 / hourlyDelta)):
				start = start_day + datetime.timedelta(hours=24 * i + j * hourlyDelta)
				stop = start_day + datetime.timedelta(hours=24 * i + (j + 1) * hourlyDelta)

				if start.hour < 8 or start.hour > 18:
					continue

				hourlyChangeData.append(vars(self.da_obj.returnVolumeSummary(start, stop)))
				hourlyChangeData[-1]['Day'] = i + 1
				hourlyChangeData[-1]['Midpoint'] = i + 1 + ((j + 0.5) * hourlyDelta) / 24
				hourlyChangeData[-1]['StartTime'] = str(start)

				current_ax = figHourly.add_subplot(gridHourly[i, current_j])

				current_ax.imshow(self.da_obj.returnHeightChange(start, stop, cropped=True), vmin=-v, vmax=v)
				current_ax.set_adjustable('box')
				current_ax.tick_params(colors=[0, 0, 0, 0])
				if i == 0:
					current_ax.set_title(str(j * hourlyDelta) + '-' + str((j + 1) * hourlyDelta))
				current_j += 1

			current_ax = figHourly.add_subplot(gridHourly[i, -2])
			current_ax.imshow(self.da_obj.returnBowerLocations(stop - datetime.timedelta(hours=24), stop, cropped=True),
							  vmin=-v, vmax=v)
			current_ax.set_adjustable('box')
			current_ax.tick_params(colors=[0, 0, 0, 0])
			if i == 0:
				current_ax.set_title('Daily\nMask')

			current_ax = figHourly.add_subplot(gridHourly[i, -1])
			current_ax.imshow(self.da_obj.returnHeightChange(stop - datetime.timedelta(hours=24), stop, cropped=True),
							  vmin=-v, vmax=v)
			current_ax.set_adjustable('box')
			current_ax.tick_params(colors=[0, 0, 0, 0])
			if i == 0:
				current_ax.set_title('Daily\nChange')

		totalDT = pd.DataFrame([totalChangeData])
		dailyDT = pd.DataFrame(dailyChangeData)
		hourlyDT = pd.DataFrame(hourlyChangeData)

		writer = pd.ExcelWriter(self.fileManager.localDepthSummaryFile)
		totalDT.to_excel(writer, 'Total')
		dailyDT.to_excel(writer, 'Daily')
		hourlyDT.to_excel(writer, 'Hourly')
		writer.save()

		bottomGrid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gridDaily[2], hspace=0.05)
		bIAx = figDaily.add_subplot(bottomGrid[1])
		bIAx.axhline(linewidth=1, alpha=0.5, y=0)
		bIAx.scatter(dailyDT['Midpoint'], dailyDT['depthBowerIndex'])
		bIAx.scatter(hourlyDT['Midpoint'], hourlyDT['depthBowerIndex'])
		bIAx.set_xlabel('Day')
		bIAx.set_ylabel('Bower\nIndex')
		bIAx.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

		volAx = figDaily.add_subplot(bottomGrid[0], sharex=bIAx)
		volAx.plot(dailyDT['Midpoint'], dailyDT['depthBowerVolume'])
		volAx.plot(hourlyDT['Midpoint'], hourlyDT['depthBowerVolume'])
		volAx.set_ylabel('Volume\nChange')
		plt.setp(volAx.get_xticklabels(), visible=False)

		figDaily.savefig(self.fileManager.localDailyDepthSummaryFigure)
		figHourly.savefig(self.fileManager.localHourlyDepthSummaryFigure)

		plt.close('all')

	def createRGBVideo(self):
		rawDepthData = np.load(self.fileManager.localRawDepthFile)
		smoothDepthData = np.load(self.fileManager.localSmoothDepthFile)
		cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
		cmap.set_bad(color = 'black')

		median_height = np.nanmedian(smoothDepthData)

		for i, frame in enumerate(self.fileManager.lp.frames):

			if i==0:
				outMovie = skvideo.io.FFmpegWriter(self.fileManager.localRGBDepthVideo)
				#outMovie = cv2.VideoWriter(self.fileManager.localRGBDepthVideo, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (depthRGB.shape[1],depthRGB.shape[0]))
			picture = plt.imread(self.fileManager.localProjectDir + frame.pic_file)
			#plt.text(x, y, s, bbox=dict(fill=False, edgecolor='red', linewidth=2))
			first_smooth_depth_cmap = cmap(plt.Normalize(-5, 5)(smoothDepthData[i] - smoothDepthData[0]))
			first_raw_depth_cmap = cmap(plt.Normalize(-5, 5)(rawDepthData[i] - rawDepthData[0]))
			outMovie.writeFrame(np.hstack([picture,first_raw_depth_cmap[:,:,0:3]*255,first_smooth_depth_cmap[:,:,0:3]*255]))

		outMovie.close()

