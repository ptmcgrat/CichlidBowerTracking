
import matplotlib.pyplot as plt
import matplotlib
from helper_modules.depth_analyzer import DepthAnalyzer as DA
from helper_modules.depth_analyzer import ClusterAnalyzer as DA

import pandas as pd 

warnings.filterwarnings('ignore')



class SummaryPreparer:
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, workers = None):
		
		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.device = self.fileManager.lp.device
		self.lp = self.fileManager.lp
		#self.createLogFile()

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localSummaryDir)

		self.fileManager.downloadData(self.fileManager.localLogfile)
		self.fileManager.downloadData(self.fileManager.localDepthCropFile)
		self.fileManager.downloadData(self.fileManager.localVideoCropFile)
		self.fileManager.downloadData(self.fileManager.localSmoothDepthFile)
		self.fileManager.downloadData(self.fileManager.localTransMFile)
		self.fileManager.downloadData(self.fileManager.localAllLabeledClustersFile)

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localLogfile)
		assert os.path.exists(self.fileManager.localDepthCropFile)
		assert os.path.exists(self.fileManager.localSmoothDepthFile)
		assert os.path.exists(self.fileManager.localTransMFile)
		assert os.path.exists(self.fileManager.localAllLabeledClustersFile)

	def uploadProjectData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.finalSummaryFigure)
		if delete:
			shutil.rmtree(self.fileManager.localProjectDir)

	def createLogFile(self):
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		with open(self.fileManager.localSummaryLogfile,'w') as f:
			print('GitBranch: ' + self.fileManager.branch_name, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

			output = subprocess.run(['conda','list'], capture_output = True)
			print(output.stdout.decode('utf-8'), file = f)


	def createSummaryFigures(self):

		# Create all figures based on depth data. Adjust hourlyDelta to influence the resolution of the
		# HourlyDepthSummary.pdf figure

		# Check that the DepthAnalzer object has been created, indicating that the required files are present.
		# Otherwise, skip creation of Depth Figures
		self.da_obj = DA(self.fileManager)
		self.cl_obj = CA(self.fileManager)
		pdb.set_trace()
		for i,trial in enumerate(self.lp.trials):
			localTrialFigureFile = self.summaryDir + trial.figureFile
			num_days = len(trial.days)
			figTrial, ax = plt.subplots(nrows = 10, ncols = num_days, figsize=(40, num_days*3))
			figTrial.suptitle(self.lp.projectID + ' Trial ' + str(i+1) + ' Summary File')

			for j, (first_frame,last_frame) in enumerate(reversed(trial.days)):
		

				current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2]]
				ax[i,0].imshow(self.da_obj.returnHeightChange(start_frame.time, last_frame.time, cropped=True), vmin=-v, vmax=v)
				bowerVolume = self.da_obj.returnVolumeSummary(first_frame.time,last_frame.time).depthBowerVolume
				current_axs[0].set_title(str(first_frame.time.month) + '/' + str(first_frame.time.day) + ':' + trial.days_videos[trial.num_days - j - 1])
				current_axs[i,1].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, cropped=True), vmin=-v/2, vmax=v/2)
				current_axs[i,2].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, masked=True, cropped=True), vmin=-v/2, vmax=v/2)
				[ax.tick_params(colors=[0, 0, 0, 0]) for ax in current_axs]
				[ax.set_adjustable('box') for ax in current_axs]

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


			cax = figDaily.add_subplot(midGrid[:, -1])
			plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)
			current_grid_idx += 1

		hourly_dt['NewTime'] = [x.hour + 0.5 for x in hourly_dt.Time]
		bottomGrid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gridDaily[-1], hspace=0.05)
		bIAx = figDaily.add_subplot(bottomGrid[0])
		bIAx.axhline(linewidth=1, alpha=0.5, y=0)
		bIAx.scatter(hourly_dt['NewTime'], hourly_dt['Volume'])
		bIAx.set_xlabel('Hour')
		bIAx.set_ylabel('Volume (cm3)')
		figDaily.savefig(self.fileManager.localDailyDepthSummaryFigure)
		plt.close('all')


