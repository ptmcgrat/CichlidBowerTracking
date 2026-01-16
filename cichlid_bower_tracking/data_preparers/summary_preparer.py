import os, pdb, datetime, shutil
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from helper_modules.depth_analyzer import DepthAnalyzer as DA
from helper_modules.depth_analyzer import ClusterAnalyzer as CA
import pandas as pd 
import PyPDF2 as pypdf

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

	def uploadData(self, delete = True):
		self.fileManager.uploadData(self.fileManager.localSummarizedClustersEvents)
		self.fileManager.uploadData(self.fileManager.localSummarizedBuildingFigure)
		self.fileManager.uploadData(self.fileManager.localSummarizedHourlyClusterFigure)
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
		cat_h_dt = pd.DataFrame(columns = ['ProjectID','Trial#','Day','Hour','ManipulationID','Number'])
		com_h_dt = pd.DataFrame(columns = ['ProjectID','Trial#','Day','Hour','ManipulationGroup','Number'])
		for i,trial in enumerate(self.lp.trials):
			localTrialFigureFile = self.fileManager.localSummaryDir + trial.figureFile
			num_days = len(trial.days)
			figTrial, axes = plt.subplots(nrows = 15, ncols = num_days, figsize=(num_days, 15), squeeze=False)
			figTrial.suptitle(self.lp.projectID + ' Trial ' + str(i+1) + ' Summary File')
			start_frame = trial.days[0][0]

			for j, (first_frame,last_frame) in enumerate(trial.days):
				day_stamp = first_frame.time.replace(hour = 0, minute=0, second=0, microsecond=0)

				#current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2]]
				axes[0,j].imshow(self.da_obj.returnHeightChange(start_frame.time, last_frame.time, cropped=True), vmin=-v, vmax=v)
				axes[0,j].set_title('Day ' + str(j+1))
				#bowerVolume = self.da_obj.returnVolumeSummary(first_frame.time,last_frame.time).depthBowerVolume
				#current_axs[0].set_title(str(first_frame.time.month) + '/' + str(first_frame.time.day) + ':' + trial.days_videos[trial.num_days - j - 1])
				axes[1,j].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, cropped=True), vmin=-v/2, vmax=v/2)
				#axes[2,j].imshow(self.da_obj.returnHeightChange(first_frame.time, last_frame.time, masked=True, cropped=True), vmin=-v/2, vmax=v/2)
				# k = 2 Scoops plus spits
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'c')
				y = self.da_obj.height - y
				axes[2,j].scatter(x,y,s = 0.05, color = 'blue')
				x2,y2 = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'p')
				y2 = self.da_obj.height - y2
				axes[2,j].scatter(x2,y2,s = 0.05, color = 'orange')
				axes[2,j].set_xlim(0,self.da_obj.width)
				axes[2,j].set_ylim(0,self.da_obj.height)
				axes[2,j].set_title('Events: ' + str(len(x) + len(x2)), fontsize = 6)

				# k = 2 Scoops plus spits (high confidence)
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'c', confidence = 0.75)
				y = self.da_obj.height - y
				axes[3,j].scatter(x,y,s = 0.05, color = 'blue')
				x2,y2 = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'p', confidence = 0.75)
				y2 = self.da_obj.height - y2
				axes[3,j].scatter(x2,y2,s = 0.05, color = 'orange')
				axes[3,j].set_xlim(0,self.da_obj.width)
				axes[3,j].set_ylim(0,self.da_obj.height)
				axes[3,j].set_title('Events: ' + str(len(x) + len(x2)), fontsize = 6)
				# k = 2 Scoops plus spits (day time)
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'c', peak_bower = True)
				y = self.da_obj.height - y
				axes[4,j].scatter(x,y,s = 0.05, color = 'blue')
				x2,y2 = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, 'p', peak_bower = True)
				y2 = self.da_obj.height - y2
				axes[4,j].scatter(x2,y2,s = 0.05, color = 'orange')
				axes[4,j].set_xlim(0,self.da_obj.width)
				axes[4,j].set_ylim(0,self.da_obj.height)
				axes[4,j].set_title('Events: ' + str(len(x) + len(x2)), fontsize = 6)
				
				# k = 2 Scoops plus spits (day time)
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, ['f','t','m'])
				y = self.da_obj.height - y
				axes[5,j].scatter(x,y,s = 0.05, color = 'blue')
				axes[5,j].set_xlim(0,self.da_obj.width)
				axes[5,j].set_ylim(0,self.da_obj.height)
				axes[5,j].set_title('Events: ' + str(len(x)), fontsize = 6)

				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, ['f','t','m'], peak_bower = True)
				y = self.da_obj.height - y
				axes[6,j].scatter(x,y,s = 0.05, color = 'blue')
				axes[6,j].set_xlim(0,self.da_obj.width)
				axes[6,j].set_ylim(0,self.da_obj.height)
				axes[6,j].set_title('Events: ' + str(len(x)), fontsize = 6)

				for k,bid in enumerate(['c', 'p', 'b', 'f', 't', 'm', 's', 'd','o','x']):
					x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, bid)
					y = self.da_obj.height - y

					axes[k+7,j].scatter(x,y,s = 0.05)
					axes[k+7,j].set_xlim(0,self.da_obj.width)
					axes[k+7,j].set_ylim(0,self.da_obj.height)
					axes[k+7,j].set_title('Events: ' + str(len(x)), fontsize = 6)
					if j == 0:
						axes[k+7,j].set_ylabel(self.cl_obj.bid_labels[bid])
					e_dt.loc[len(e_dt)] = [self.lp.projectID, 'Trial_' + str(i+1), j, self.cl_obj.bid_labels[bid], len(x)]
				
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, None, False, True)
				axes[13,j].scatter(x,y,s = 0.05)
				axes[13,j].set_xlim(0,self.da_obj.width)
				axes[13,j].set_ylim(0,self.da_obj.height)
				axes[13,j].set_title('Events: ' + str(len(x)), fontsize = 6)
				x,y = self.cl_obj.returnDepthCoordinates(first_frame.time, last_frame.time, None, None, False)
				axes[14,j].scatter(x,y,s = 0.05)
				axes[14,j].set_xlim(0,self.da_obj.width)
				axes[14,j].set_ylim(0,self.da_obj.height)
				axes[14,j].set_title('Events: ' + str(len(x)), fontsize = 6)
				if j==0:
					axes[0,j].set_ylabel('Total depth')
					axes[1,j].set_ylabel('Daily depth')
					axes[2,j].set_ylabel('All Build')
					axes[3,j].set_ylabel('HC Build')
					axes[4,j].set_ylabel('Peak Build')
					axes[5,j].set_ylabel('All Feed')
					axes[6,j].set_ylabel('Peek Feed')
					
					#axes[2,j].set_ylabel('Daily bower')
					axes[13,j].set_ylabel('Cropped clips')
					axes[14,j].set_ylabel('Not created')

				for hour in range(8,20):
					start = day_stamp + datetime.timedelta(hours=hour)
					stop = day_stamp + datetime.timedelta(hours=hour+1)
					if stop < first_frame.time or start > last_frame.time:
						continue
					output, combined_output = self.cl_obj.returnCategoryCounts(start, stop)
					for bid in ['c', 'p', 'b', 'f', 't', 'm', 's']:
						cat_h_dt.loc[len(cat_h_dt)] = [self.lp.projectID, 'Trial_' + str(i+1), j, hour, self.cl_obj.bid_labels[bid], output[bid]]
					for cat in combined_output.keys():
						com_h_dt.loc[len(com_h_dt)] = [self.lp.projectID, 'Trial_' + str(i+1), j, hour, cat, combined_output[cat]]
				[ax.set_xticks([]) for ax in axes[:,j]]
				[ax.set_yticks([]) for ax in axes[:,j]]
				#[ax.set_adjustable('box') for ax in axes[:,j]]
			
			figTrial.tight_layout()
			#plt.show()
			figTrial.savefig(localTrialFigureFile)
		plt.close('all')

		figHourly, axes = plt.subplots(nrows = len(self.lp.trials), ncols = 4, figsize=(12, 3*len(self.lp.trials)), squeeze=False)
		for i,trial in enumerate(self.lp.trials):
			for j,cat in enumerate(com_h_dt['ManipulationGroup'].unique()):
				sub_dt = com_h_dt[(com_h_dt['Trial#'] == 'Trial_'+str(i+1)) & (com_h_dt.ManipulationGroup == cat)]
				sns.boxplot(x='Hour', y='Number', data=sub_dt, ax = axes[i,j], showfliers = False)
				sns.stripplot(x='Hour', y='Number', data=sub_dt, color=".25", size=3, ax=axes[i,j])
				if i == 0:
					axes[i,j].set_title(cat)
				if j == 0:
					axes[i,j].set_ylabel('Trial_'+str(i+1))
		figHourly.tight_layout()
		figHourly.savefig(self.fileManager.localSummarizedHourlyClusterFigure)
		e_dt.to_csv(self.fileManager.localSummarizedClustersEvents)
		plt.close('all')

		writer = pypdf.PdfWriter()
		for i,trial in enumerate(self.lp.trials):
			localTrialFigureFile = self.fileManager.localSummaryDir + trial.figureFile
			f = open(localTrialFigureFile, 'rb')
			reader = pypdf.PdfReader(f)
			for page_number in range(len(reader.pages)):
				writer.add_page(reader.pages[page_number])
		with open(self.fileManager.localSummarizedBuildingFigure, 'wb') as f:
			writer.write(f)



