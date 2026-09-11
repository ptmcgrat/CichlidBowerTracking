
import scipy.signal
import skvideo.io
import numpy as np
import pdb, os, sys, datetime, warnings, copy, subprocess, shutil, io
import matplotlib.pyplot as plt
from matplotlib import path as mpl_path
from helper_modules.depth_masking import trialChurnMask
from helper_modules.depth_interpolation import interpolate_time, interpolate_space
import matplotlib
from PIL import Image,ImageDraw
from helper_modules.depth_analyzer import DepthAnalyzer as DA
from matplotlib import (cm, colors, gridspec, ticker)
import pandas as pd 
from skimage import morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helper_modules.depth_interpolation import process_day
from helper_modules.depth_endpoints import saveDailyEndpoints
warnings.filterwarnings('ignore')



class DepthPreparer:
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, picture = True, workers = None):
		
		self.__version__ = '1.0.0'
		self.picture = picture
		self.fileManager = fileManager
		self.device = self.fileManager.lp.device
		self.lp = self.fileManager.lp
		self.createLogFile()

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
		self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		self.fileManager.createDirectory(self.fileManager.localSummaryDir)

		self.fileManager.downloadData(self.fileManager.localLogfile)
		self.fileManager.downloadData(self.fileManager.localFrameDir, tarred = True)
		#self.fileManager.downloadData(self.fileManager.localSmoothDepthFile)
		self.fileManager.downloadData(self.fileManager.localDepthCropFile)
		if self.picture:
			self.fileManager.downloadData(self.fileManager.localBuildPhotosDir)

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

	def uploadProjectData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localSmoothDepthFile)
		self.fileManager.uploadData(self.fileManager.localRawDepthFile)
		self.fileManager.uploadData(self.fileManager.localProjectDir + 'DepthFiles')
		#self.fileManager.uploadData(self.fileManager.localSmoothDepthDT)

		#self.fileManager.uploadData(self.fileManager.localRGBDepthVideo)
		self.fileManager.uploadData(self.fileManager.localDepthLogfile)
		self.fileManager.uploadData(self.fileManager.localDailyDepthSummaryFigure)
		self.fileManager.uploadData(self.fileManager.localDepthSummaryFile)
		#self.fileManager.uploadData(self.fileManager.localHourlyDepthSummaryFigure)

			#self.uploadData(self.localPaceDir)
		if delete:
			shutil.rmtree(self.fileManager.localProjectDir)

	def createLogFile(self):
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		with open(self.fileManager.localDepthLogfile,'w') as f:
			print('GitBranch: ' + self.fileManager.branch_name, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

			output = subprocess.run(['conda','list'], capture_output = True)
			print(output.stdout.decode('utf-8'), file = f)


	def createSmoothedArray(self, goodDataCutoff = 0.7, tunits = 71, order = 4, max_depth = 4, max_height = 8):
		
		# Create arrays to store raw depth data and data in the daytime
		rawDepthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
		rawStdData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
		
		#daytimeData = np.empty(shape = (sum([x.lof for x in self.lp.frames]), self.lp.height, self.lp.width))

		# Read in each frame and store it. Also keep track of the indeces that are in the daytime
		#day_idx = 0
		#day_start_stop = OrderedDict() # Dictionary to hold first and last frame indeces for good and bad data for each day

		#ad_idx = 0
		for i, frame in enumerate(self.lp.frames):
			try:
				data = np.load(self.fileManager.localProjectDir + frame.npy_file)
				data_std = np.load(self.fileManager.localProjectDir + frame.std_file)
			except (FileNotFoundError,EOFError):
				print('Bad frame: ' + str(i) + ', ' + frame.npy_file)
				rawDepthData[i] = rawDepthData[i-1]
				rawStdData[i] = rawStdData[i-1]
			else:
				rawDepthData[i] = data
				rawStdData[i] = data_std

		crop = eval('[' + open(self.fileManager.localDepthCropFile).read() + ']')
		gy, gx = np.mgrid[0:self.lp.height, 0:self.lp.width]
		pts = np.column_stack([gx.ravel(), gy.ravel()])
		self.tray_mask = mpl_path.Path(crop).contains_points(pts).reshape(self.lp.height, self.lp.width)

		# Save raw data file
		np.save(self.fileManager.localRawDepthFile, rawDepthData)

		# Interpolate missing data
		# Make copy of raw data
		interpDepthData = rawDepthData.copy()

		for trial in self.lp.trials:
			for start_f, stop_f in trial.days:          # pass one: temporal
				day = interpDepthData[start_f.index:stop_f.index + 1]
				day[:] = interpolate_time(day, min_good=goodDataCutoff)

			mask, stats = trialChurnMask(interpDepthData, self.lp, trial,
										 tray_mask=self.tray_mask, k=5,
										 std_mean=trial_std_mean)
			if stats['applied']:
				interpDepthData[trial_start:trial_stop][:, mask] = np.nan

			for start_f, stop_f in trial.days:          # pass two: spatial, fills the mask
				day = interpDepthData[start_f.index:stop_f.index + 1]
				day[:] = interpolate_space(day, usable=self.tray_mask)
		
		# Save interpolated data
		np.save(self.fileManager.localSmoothDepthFile, interpDepthData)
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
		#non_nans = np.count_nonzero(~np.isnan(daytimeData), axis = 0)
		#smoothDepthData[:,non_nans < minimumGoodData*daytimeData.shape[0]] = np.nan

		# Filter out data with bad standard deviations
		#stds = np.nanstd(daytimeData, axis = 0)
		#smoothDepthData[:,stds > 1.5] = np.nan # Filter out data with std > 1.5 cm

		# Filter out data that is too close or too far from the sensor
		#average_depth = np.nanmean(daytimeData, axis = 0)
		#median_height = np.nanmedian(average_depth)
		#smoothDepthData[:,(average_depth > median_height + max_depth) | (average_depth < median_height - max_height)] = np.nan # Filter out data 4cm lower and 8cm higher than tray

		# Nighttime data is bad. Set it to average of data before and after.
		for i,frame in enumerate(self.lp.frames):
			if not frame.lof:
				try:
					smoothDepthData[i] = np.nanmean((smoothDepthData[frame.nearest_day[0].index],smoothDepthData[frame.nearest_day[1].index]), axis = 0)
				except IndexError:
					pdb.set_trace()
				except AttributeError:
					if frame.time > self.lp.trials[-1].stopTime:
						continue
					else:
						continue
						pdb.set_trace()
		# Smooth data with savgol_filter
		np.save(self.fileManager.localSmoothDepthFile, smoothDepthData)
		saveDailyEndpoints(self.fileManager, self.lp, rawDepthData, smoothDepthData, rawStdData)

	def createDepthFigures(self, hourlyDelta=2):

		# Create all figures based on depth data. Adjust hourlyDelta to influence the resolution of the
		# HourlyDepthSummary.pdf figure

		# Check that the DepthAnalzer object has been created, indicating that the required files are present.
		# Otherwise, skip creation of Depth Figures
		self.da_obj = DA(self.fileManager)

		num_trials = self.lp.num_trials
		total_rows = sum([4*x.num_rows for x in self.lp.trials]) + 2*num_trials

		# figures based on the depth data
		# Create summary figure of daily values
		figDaily = plt.figure(num=1, figsize=(11, 1.5*total_rows))
		figDaily.suptitle(self.lp.projectID + ' Daily Depth Summary')
		gridDaily = gridspec.GridSpec(1, 1)
		#print(num_trials + total_rows + 1)
		daily_dt = pd.DataFrame(columns = ['ProjectID','Trial_ID','Day','Threshold','CastleVolume','PitVolume'])
		masterGrid = gridspec.GridSpecFromSubplotSpec(total_rows, 6, subplot_spec=gridDaily[0])
		current_row = 0
		
		for i,trial in enumerate(reversed(self.lp.trials)):
			

			start_frame = trial.daylight_frames[0]
			last_frame = trial.daylight_frames[-1]
			reset_frame = trial.reset_frame
			#totalChangeData = vars(self.da_obj.returnVolumeSummary(self.lp.frames[start_index].time, self.lp.frames[last_index].time))

			topAx1 = figDaily.add_subplot(masterGrid[current_row:current_row+2,0:2])
			# Show picture of total depth change
			try:
				if self.picture:
					build_photo = self.fileManager.localBuildPhotosDir + self.fileManager.lp.sampleID + '_TR_Trial' + str(i+1) + '.jpg'
					build_rgb = plt.imread(build_photo)
					topAx1_ax = topAx1.imshow(build_rgb)
				else:
					topAx1_ax = topAx1.imshow(self.da_obj.returnHeightChange(
						start_frame.time, last_frame.time, cropped=False), vmin=-2, vmax=2)
					bowerVolume = self.da_obj.returnVolumeSummary(start_frame.time,last_frame.time).depthBowerVolume
					topAx1.set_title('Total Depth Change (' + str(int(bowerVolume)) + 'cm3)')
					divider = make_axes_locatable(topAx1)
					cax1 = divider.append_axes("right", size="5%", pad=0.05)
					plt.colorbar(topAx1_ax, cax=cax1)

				topAx1.tick_params(colors=[0, 0, 0, 0])

				
			except FileNotFoundError:
				pass
			# Show picture of reset depth change
			topAx2 = figDaily.add_subplot(masterGrid[current_row:current_row+2,2:4])
			topAx2_ax = topAx2.imshow(self.da_obj.returnHeightChange(reset_frame.time, last_frame.time, cropped = False), vmin = -2, vmax = 2)
			bowerVolume = self.da_obj.returnVolumeSummary(reset_frame.time,last_frame.time).depthBowerVolume
			topAx2.set_title('Reset Depth Change ('+ str(int(bowerVolume)) + 'cm3)')
			topAx2.tick_params(colors=[0, 0, 0, 0])
			divider = make_axes_locatable(topAx2)
			cax2 = divider.append_axes("right", size="5%", pad=0.05)
			plt.colorbar(topAx1_ax, cax=cax2)

			# Show picture of reset depth change
			topAx3 = figDaily.add_subplot(masterGrid[current_row:current_row+2,4:6])
			data = [self.da_obj.returnVolumeSummary(start_frame.time,last_frame.time,thresh = x) for x in [.1,.5,1,1.5,2,2.5,3]]
			topAx3.plot([.1,.5,1,1.5,2,2.5,3],[x.depthCastleVolume for x in data], '-o', color = 'gold', label = 'Castle volume')
			topAx3.plot([.1,.5,1,1.5,2,2.5,3],[x.depthPitVolume for x in data], '-o', color = 'blue', label = 'Pit volume')
			topAx3.set_title('Pit/castle volume by threshold')
			#topAx3.set_xticklabels([]) 
			#topAx3.set_figheight(3)

			#day_info = self.depth_dt[(self.depth_dt.DaytimeData == True)&(self.depth_dt.Trial == 'Trial_' + str(i))].groupby('RelativeDay').agg(day_start = ('Index','first'), day_stop = ('Index','last')).sort_index(ascending = False)

			num_days = min(trial.num_days,6)

			v = 2.0
			current_row += 2
			offset = 0
			for j, (first_frame,last_frame) in enumerate(reversed(trial.days)):
				if j % 6 == 0:
					if j!=0:
						current_row += 4
						offset += 6

				totalAx = figDaily.add_subplot(masterGrid[current_row,-1*(j-offset+1)])
				changeAx = figDaily.add_subplot(masterGrid[current_row + 1,-1*(j-offset+1)])
				bowerAx = figDaily.add_subplot(masterGrid[current_row + 2,-1*(j-offset+1)])
				threshAx = figDaily.add_subplot(masterGrid[current_row + 3,-1*(j-offset+1)])
				
				totalAx.imshow(self.da_obj.returnHeightChange(start_frame.time, last_frame.time, cropped=True), vmin=-v, vmax=v)
				bowerVolume = self.da_obj.returnVolumeSummary(first_frame.time,last_frame.time).depthBowerVolume
				totalAx.set_title(str(first_frame.time.month) + '/' + str(first_frame.time.day) + ':' + trial.days_videos[trial.num_days - j - 1])
				changeAx.imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, cropped=True), vmin=-v, vmax=v)
				bowerAx.imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, masked=True, cropped=True), vmin=-1, vmax=1)
				[ax.tick_params(colors=[0, 0, 0, 0]) for ax in [totalAx,changeAx,bowerAx]]
				[ax.set_adjustable('box') for ax in [totalAx,changeAx,bowerAx]]

				data = [self.da_obj.returnVolumeSummary(first_frame.time,last_frame.time,thresh = x) for x in [.1,.5,1,1.5,2,2.5,3]]
				threshAx.plot([.1,.5,1,1.5,2,2.5,3],[x.depthCastleVolume for x in data], '-o', color = 'gold', label = 'Castle volume')
				threshAx.plot([.1,.5,1,1.5,2,2.5,3],[x.depthPitVolume for x in data], '-o', color = 'blue', label = 'Pit volume')
				threshAx.set_ylim([0,250])
				daily_dt.loc[len(daily_dt)] = [self.lp.projectID,trial.number, trial.days_videos[trial.num_days - j - 1], 0.5, data[1].depthCastleVolume, data[1].depthPitVolume]
				daily_dt.loc[len(daily_dt)] = [self.lp.projectID,trial.number, trial.days_videos[trial.num_days - j - 1], 1.0, data[2].depthCastleVolume, data[2].depthPitVolume]
				daily_dt.loc[len(daily_dt)] = [self.lp.projectID,trial.number, trial.days_videos[trial.num_days - j - 1], 1.5, data[3].depthCastleVolume, data[3].depthPitVolume]

				#threshAx.set_xticklabels([]) 
			if j % 6 != 1:
				current_row += 4

				"""
				#good_data_start = self.lp.frames[day_start].time
				#good_data_stop = self.lp.frames[day_stop].time
				day_stamp = first_frame.time.replace(hour = 0, minute=0, second=0, microsecond=0)
				for k in range(8,20):
					start = day_stamp + datetime.timedelta(hours=k)
					stop = day_stamp + datetime.timedelta(hours=k+1)
					if start < first_frame.time or stop > last_frame.time:
						continue
					volume = self.da_obj.returnVolumeSummary(start,stop).depthBowerVolume
					hourly_dt.loc[len(hourly_dt.index)] = ['Trial_' + str(num_trials - i),start.replace(minute = 30),volume]
				"""


		"""
		hourly_dt['NewTime'] = [x.hour + 0.5 for x in hourly_dt.Time]
		bottomGrid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gridDaily[-1], hspace=0.05)
		bIAx = figDaily.add_subplot(bottomGrid[0])
		bIAx.axhline(linewidth=1, alpha=0.5, y=0)
		bIAx.scatter(hourly_dt['NewTime'], hourly_dt['Volume'])
		bIAx.set_xlabel('Hour')
		bIAx.set_ylabel('Volume (cm3)')
		"""
		daily_dt.to_csv(self.fileManager.localDepthSummaryFile)
		plt.tight_layout()
		figDaily.savefig(self.fileManager.localDailyDepthSummaryFigure)
		plt.close('all')
		

	def createRGBVideo(self):
		lp = self.fileManager.lp
		rawDepthData = np.load(self.fileManager.localRawDepthFile)
		smoothDepthData = np.load(self.fileManager.localSmoothDepthFile)
		#cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
		#cmap.set_bad(color = 'black')

		median_height = np.nanmedian(smoothDepthData)
		matplotlib.use("Agg")



		for i, frame in enumerate(self.fileManager.lp.frames):

			trials = [(j+1,x) for (j,x) in enumerate(lp.trials) if frame.time >= x.startTime and frame.time <= x.stopTime]
			trial = str(trials[0][0]) if len(trials) == 1 else 'None'
			fig = plt.figure(figsize=(8, 8))
			fig.suptitle('Trial ' + trial + ': ' + str(frame.time.ctime()))
			ax1 = fig.add_subplot(2,2,1)       
			ax2 = fig.add_subplot(2,2,2)
			ax3 = fig.add_subplot(2,2,3)
			ax4 = fig.add_subplot(2,2,4)

			if i==0:
				outMovie = skvideo.io.FFmpegWriter(self.fileManager.localRGBDepthVideo, outputdict={'-vcodec': 'libx264'})
				#outMovie = cv2.VideoWriter(self.fileManager.localRGBDepthVideo, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (depthRGB.shape[1],depthRGB.shape[0]))
			if os.path.exists(self.fileManager.localProjectDir + frame.pic_file):
				depthRGB = plt.imread(self.fileManager.localProjectDir + frame.pic_file)
			else:
				print('Cant find ' + frame.pic_file + '. Using previous')
			if len(trials) == 1:
				start_index = trials[0][1].frames[0].index
			else:
				start_index = 0
			ax1.imshow(depthRGB, cmap = 'gray')
			ax2.imshow((rawDepthData[i] - rawDepthData[start_index]),vmin=-2, vmax=2)
			ax3.imshow((rawDepthData[i]), vmin=median_height - 5, vmax=median_height + 5)
			ax4.imshow((smoothDepthData[i]), vmin=median_height - 5, vmax=median_height + 5)
			ax1.set_title('DepthRGB')
			ax2.set_title('RawDepthChange: Med=' + str(frame.med) + ',,Std=' + str(frame.std) + ',,GP=' + str(frame.gp))
			ax3.set_title('RawCurrentDepth')
			ax4.set_title('SmoothedCurrentDepth')

			fig.canvas.draw()

			# Now we can save it to a numpy array.
			data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
			data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

			#plt.text(x, y, s, bbox=dict(fill=False, edgecolor='red', linewidth=2))
			outMovie.writeFrame(data[:,:,1:4])

		outMovie.close()
		plt.close('all')

