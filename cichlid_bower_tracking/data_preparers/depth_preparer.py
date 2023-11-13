import scipy.signal
import skvideo.io
import numpy as np
import pdb, os, sys, datetime, warnings, copy, subprocess
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image,ImageDraw
from helper_modules.depth_analyzer import DepthAnalyzer as DA
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
        self.fileManager.createDirectory(self.fileManager.localMasterDir)
        self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
        self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
        self.fileManager.createDirectory(self.fileManager.localLogfileDir)
        self.fileManager.createDirectory(self.fileManager.localSummaryDir)

        self.fileManager.downloadData(self.fileManager.localLogfile)
        self.fileManager.downloadData(self.fileManager.localFrameDir, tarred = True)
        self.fileManager.downloadData(self.fileManager.localDepthCropFile)

 
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
        #self.fileManager.uploadData(self.fileManager.localSmoothDepthFile)
        self.fileManager.uploadData(self.fileManager.localSmoothDepthDT)

        #self.fileManager.uploadData(self.fileManager.localRGBDepthVideo)
        self.fileManager.uploadData(self.fileManager.localDepthLogfile)
        self.fileManager.uploadData(self.fileManager.localDailyDepthSummaryFigure)
        #self.fileManager.uploadData(self.fileManager.localHourlyDepthSummaryFigure)

            #self.uploadData(self.localPaceDir)
        if delete:
            shutil.rmtree(self.localProjectDir)

    def createLogFile(self):
        self.fileManager.createDirectory(self.fileManager.localLogfileDir)
        with open(self.fileManager.localDepthLogfile,'w') as f:
            print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
            print('NumpyVersion: ' + np.__version__, file = f)
            print('Scikit-VideoVersion: ' + skvideo.__version__, file = f)
            print('ScipyVersion: ' + scipy.__version__, file = f)
            print('Username: ' + os.getenv('USER'), file = f)
            print('Nodename: ' + os.uname().nodename, file = f)
            print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

    def createSmoothedArray(self, start_hour  = 8, stop_hour = 6, goodDataCutoff = 0.8, minimumGoodData = 0.95, tunits = 71, order = 4, max_depth = 4, max_height = 8):
        
        # Create arrays to store raw depth data and data in the daytime
        depthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
        depth_dt = pd.DataFrame(columns = ['Index','Time','DaytimeData','RelativeDay'])

        # Read in each frame and store it. Also keep track of the indeces that are in the daytime
        for i, frame in enumerate(self.lp.frames):
            try:
                data = np.load(self.fileManager.localProjectDir + frame.npy_file)

            except FileNotFoundError:
                print('Bad frame: ' + str(i) + ', ' + frame.npy_file)
                depthData[i] = depthData[i-1]
            else:
                try: 
                    #depthData[i] = data.astype('float16')
                    depthData[i] = data

                except ValueError:
                    pdb.set_trace()

            depth_dt.loc[len(depth_dt.index)] = [i,frame.time, frame.lof, (frame.time.date() - self.fileManager.dissectionTime.date()).days]

        daytime_data = depth_dt[depth_dt.DaytimeData == True].groupby('RelativeDay').agg(first_index = ('Index','first'), last_index = ('Index','last'))

        # Loop through each day and interpolate missing data, setting night time data to average of first and last frame
        night_start = 0
        for day, (start_index,stop_index) in daytime_data.iterrows():
            dailyData = depthData[start_index:stop_index] # Create view of numpy array just creating a single day during the daytime
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
            if start_index != 0: #no previous night data
                if night_start == 0:
                    depthData[night_start:start_index] = depthData[start_index]
                else:
                    depthData[night_start:start_index] = (depthData[start_index] + depthData[night_start - 1])/2
            night_start = stop_index + 1
        depthData[night_start:] = depthData[night_start-1]
        
   
        depth_dt['Trial'] = ''
        try:
            assert len(self.lp.tankresetstart) == len(self.lp.tankresetstop)
        except:
            print('Fix logfile for tankreset start and stop')

        previous_start = 0
        for i,(start_time,stop_time) in enumerate(zip(self.lp.tankresetstart,self.lp.tankresetstop)):
            depth_dt.loc[(depth_dt.Trial == '') & (depth_dt.Time < start_time),'Trial'] = 'Trial_' + str(i+1)
            depth_dt.loc[(depth_dt.Trial == '') & (depth_dt.Time > start_time) & (depth_dt.Time <= stop_time),'Trial'] = 'Trial_' + str(i+1) + '_Reset'

        # Save interpolated data
        
        # Read in manual crop and mask out data outside of crop
        """with open(self.fileManager.localDepthCropFile) as f:
            for line in f:
                depth_crop_points = eval(line.rstrip())

        img = Image.new('L', (self.lp.width, self.lp.height), 0)
        ImageDraw.Draw(img).polygon(depth_crop_points, outline=1, fill=1)
        manual_crop_mask = np.array(img)
        smoothDepthData[:,manual_crop_mask == 0] = np.nan"""

        daytimeData = depthData[depth_dt[depth_dt.DaytimeData == True].Index]
        # Mask out data with too many nans
        non_nans = np.count_nonzero(~np.isnan(daytimeData), axis = 0)
        depthData[:,non_nans < minimumGoodData*daytimeData.shape[0]] = np.nan


        # Filter out data with bad standard deviations
        stds = np.nanstd(daytimeData, axis = 0)

        depthData[:,stds > 6] = np.nan # Filter out data with std > 1.5 cm

        # Filter out data that is too close or too far from the sensor
        average_depth = np.nanmean(daytimeData, axis = 0)
        median_height = np.nanmedian(average_depth)
        depthData[:,(average_depth > median_height + max_depth) | (average_depth < median_height - max_height)] = np.nan # Filter out data 4cm lower and 8cm higher than tray


        # Smooth data with savgol_filter
        depthData = scipy.signal.savgol_filter(depthData, tunits, order, axis = 0, mode = 'mirror')
        np.save(self.fileManager.localSmoothDepthFile, depthData)
        self.depth_dt = depth_dt
        depth_dt.to_csv(self.fileManager.localSmoothDepthDT)

    def createDepthFigures(self, hourlyDelta=2):

        # Create all figures based on depth data. Adjust hourlyDelta to influence the resolution of the
        # HourlyDepthSummary.pdf figure

        # Check that the DepthAnalzer object has been created, indicating that the required files are present.
        # Otherwise, skip creation of Depth Figures
        self.da_obj = DA(self.fileManager)
        self.depth_dt = pd.read_csv(self.fileManager.localSmoothDepthDT, index_col = 0)
        self.depth_dt = self.depth_dt[~self.depth_dt.Trial.isna()]
        project_info = self.depth_dt[self.depth_dt.DaytimeData == True].groupby('Trial').agg(first_index = ('Index','first'), last_index = ('Index','last'))
        num_trials = len(self.lp.tankresetstart)
        rows = int(np.ceil((self.depth_dt[~(self.depth_dt.Trial.str.contains('Reset')) & ~(self.depth_dt.Trial == '')].groupby(['Trial']).nunique()['RelativeDay']/10)).sum())

        # figures based on the depth data
        # Create summary figure of daily values
        figDaily = plt.figure(num=1, figsize=(11, 8.5))
        figDaily.suptitle(self.lp.projectID + ' Daily Depth Summary')
        gridDaily = gridspec.GridSpec(num_trials + rows, 1)

        current_grid_idx = 0
        for i in range(1,num_trials + 1):

            start_index = project_info.loc['Trial_' + str(i)].first_index
            last_index = project_info.loc['Trial_' + str(i)].last_index
            reset_index = project_info.loc['Trial_' + str(i) + '_Reset'].last_index
            #totalChangeData = vars(self.da_obj.returnVolumeSummary(self.lp.frames[start_index].time, self.lp.frames[last_index].time))

            topGrid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gridDaily[current_grid_idx])

            # Show picture of total depth change
            topAx1 = figDaily.add_subplot(topGrid[0])
            topAx1_ax = topAx1.imshow(self.da_obj.returnHeightChange(
                self.lp.frames[start_index].time, self.lp.frames[last_index].time, cropped=False), vmin=-3, vmax=3)
            topAx1.set_title('Total Depth Change (cm)')
            topAx1.tick_params(colors=[0, 0, 0, 0])
            plt.colorbar(topAx1_ax, ax=topAx1)

            # Show picture of pit and castle mask
            topAx2 = figDaily.add_subplot(topGrid[1])
            topAx2_ax = topAx2.imshow(self.da_obj.returnHeightChange(self.lp.frames[reset_index].time, self.lp.frames[last_index].time, cropped = False), vmin = -3, vmax = 3)
            topAx2.set_title('Reset Depth Change (cm)')
            topAx2.tick_params(colors=[0, 0, 0, 0])
            plt.colorbar(topAx2_ax, ax=topAx2)

            day_info = self.depth_dt[(self.depth_dt.DaytimeData == True)&(self.depth_dt.Trial == 'Trial_' + str(i))].groupby('RelativeDay').agg(day_start = ('Index','first'), day_stop = ('Index','last')).sort_index(ascending = False)

            num_days = min(len(day_info),10)

            v = 2.0



            for j, (day,(day_start,day_stop)) in enumerate(day_info.iterrows()):
                if j % num_days == 0:
                    if j!=0:
                        cax = figDaily.add_subplot(midGrid[:, -1])
                        plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)

                    current_grid_idx += 1
                    midGrid = gridspec.GridSpecFromSubplotSpec(3, num_days + 1, subplot_spec=gridDaily[current_grid_idx])

                current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2]]
                current_axs[0].imshow(self.da_obj.returnHeightChange(self.lp.frames[day_info.iloc[-1].day_start].time, self.lp.frames[day_stop].time, cropped=True), vmin=-v, vmax=v)
                current_axs[0].set_title('%i' % (day))
                current_axs[1].imshow(self.da_obj.returnHeightChange(self.lp.frames[day_start].time, self.lp.frames[day_stop].time, cropped=True), vmin=-v, vmax=v)
                current_axs[2].imshow(self.da_obj.returnHeightChange(self.lp.frames[day_start].time, self.lp.frames[day_stop].time, masked=True, cropped=True), vmin=-v, vmax=v)
                [ax.tick_params(colors=[0, 0, 0, 0]) for ax in current_axs]
                [ax.set_adjustable('box') for ax in current_axs]
            cax = figDaily.add_subplot(midGrid[:, -1])
            plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)

            figDaily.savefig(self.fileManager.localDailyDepthSummaryFigure)

        """
        # Create summary figure of hourly values
        figHourly = plt.figure(figsize=(11, 8.5))
        gridHourly = plt.GridSpec(self.lp.numDays, int(11 / hourlyDelta) + 2, wspace=0.05, hspace=0.05)
        bounding_ax = figHourly.add_subplot(gridHourly[:, :])
        bounding_ax.xaxis.set_visible(False)
        bounding_ax.set_ylabel('Day')
        bounding_ax.set_ylim(self.lp.numDays + 0.5, 0.5)
        bounding_ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        bounding_ax.set_yticklabels(range(self.lp.numDays + 1))
        sns.despine(ax=bounding_ax, left=True, bottom=True)
        start_day = self.lp.frames[0].time.replace(hour=0, minute=0, second=0, microsecond=0)

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

        figHourly.savefig(self.fileManager.localHourlyDepthSummaryFigure)
        """
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



