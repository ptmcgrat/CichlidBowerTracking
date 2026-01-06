import os, pdb
import matplotlib.pyplot as plt
import matplotlib
from helper_modules.depth_analyzer import DepthAnalyzer as DA
from helper_modules.depth_analyzer import ClusterAnalyzer as CA
import pandas as pd 

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

	def downloadData(self):
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


	def createSummaryFigures(self, v=2.0):

		# Create all figures based on depth data. Adjust hourlyDelta to influence the resolution of the
		# HourlyDepthSummary.pdf figure

		# Check that the DepthAnalzer object has been created, indicating that the required files are present.
		# Otherwise, skip creation of Depth Figures
		self.da_obj = DA(self.fileManager)
		self.cl_obj = CA(self.fileManager)
		e_dt = pd.DataFrame(columns = ['ProjectID','Trial#','Day','ManipulationType','Number'])
		for i,trial in enumerate(self.lp.trials):
			localTrialFigureFile = self.fileManager.localSummaryDir + trial.figureFile
			num_days = len(trial.days)
			figTrial, axes = plt.subplots(nrows = 10, ncols = num_days, figsize=(num_days, 10))
			figTrial.suptitle(self.lp.projectID + ' Trial ' + str(i+1) + ' Summary File')
			start_frame = trial.days[0][0]

			for j, (first_frame,last_frame) in enumerate(trial.days):

				#current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2]]
				axes[0,j].imshow(self.da_obj.returnHeightChange(start_frame.time, last_frame.time, cropped=True), vmin=-v, vmax=v)
				axes[0,j].set_title('Day ' + str(j+1))
				#bowerVolume = self.da_obj.returnVolumeSummary(first_frame.time,last_frame.time).depthBowerVolume
				#current_axs[0].set_title(str(first_frame.time.month) + '/' + str(first_frame.time.day) + ':' + trial.days_videos[trial.num_days - j - 1])
				axes[1,j].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, cropped=True), vmin=-v/2, vmax=v/2)
				axes[2,j].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, masked=True, cropped=True), vmin=-v/2, vmax=v/2)
				if j==0:
					axes[0,j].set_ylabel('Total depth')
					axes[1,j].set_ylabel('Daily depth')
					axes[2,j].set_ylabel('Daily bower')

				for k,bid in enumerate(['c', 'p', 'b', 'f', 't', 'm', 's']):
					x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, bid)
					y = self.da_obj.height - y
					axes[k+3,j].scatter(x,y,s = 0.05)
					axes[k+3,j].set_xlim(0,self.da_obj.width)
					axes[k+3,j].set_ylim(0,self.da_obj.height)
					axes[k+3,j].set_title('Events: ' + str(len(x)), fontsize = 6)
					if j == 0:
						axes[k+3,j].set_ylabel(self.cl_obj.bid_labels[bid])
					e_dt.loc[len(e_dt)] = [self.lp.projectID, 'Trial_' + str(i+1), j, self.cl_obj.bid_labels[bid], len(x)]
				[ax.set_xticks([]) for ax in axes[:,j]]
				[ax.set_yticks([]) for ax in axes[:,j]]
				#[ax.set_adjustable('box') for ax in axes[:,j]]
			
			figTrial.tight_layout()
			plt.show()
			figTrial.savefig(localTrialFigureFile)
		e_dt.to_csv(self.fileManager.localSummarizedClustersEvents)

		plt.close('all')


