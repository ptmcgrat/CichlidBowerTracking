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
        self.fileManager.uploadData(self.fileManager.localSmoothDepthFile)
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

    def createSmoothedArray(self, goodDataCutoff = 0.7, minimumGoodData = 0.7, std_cutoff = 0.25, max_depth = 4, max_height = 8):
        
        # Create arrays to store raw depth data and data in the daytime
        depthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
        depth_dt = pd.DataFrame(columns = ['Index','Time','DaytimeData','RelativeDay','BadStdPixels','LoadedBadPixels'])

        # Read in each frame and store it. Also keep track of the indeces that are in the daytime
        for i, frame in enumerate(self.lp.frames):
            try:
                data = np.load(self.fileManager.localProjectDir + frame.npy_file)
                std = np.load(self.fileManager.localProjectDir + frame.std_file)
                data[std > std_cutoff] = np.nan

            except FileNotFoundError:
                print('Bad frame: ' + str(i) + ', ' + frame.npy_file)
                depthData[i] = depthData[i-1]
            else:
                depthData[i] = data

            depth_dt.loc[len(depth_dt.index)] = [i,frame.time, frame.lof, (frame.time.date() - self.fileManager.dissectionTime.date()).days,np.sum(std>std_cutoff), np.sum(np.isnan(data))]

        
        # Divide into trials based on
        depth_dt['Trial'] = ''
        try:
            assert len(self.lp.tankresetstart) == len(self.lp.tankresetstop)
        except:
            print('Fix logfile for tankreset start and stop')

        previous_start = 0
        for i,(start_time,stop_time) in enumerate(zip(self.lp.tankresetstart,self.lp.tankresetstop)):
            depth_dt.loc[(depth_dt.Trial == '') & (depth_dt.Time < start_time - datetime.timedelta(minutes = 5)),'Trial'] = 'Trial_' + str(i+1)
            depth_dt.loc[(depth_dt.Trial == '') & (depth_dt.Time > start_time - datetime.timedelta(minutes = 5)) & (depth_dt.Time <= stop_time + datetime.timedelta(minutes = 5)),'Trial'] = 'Trial_' + str(i+1) + '_Reset'

        # Loop through each day and interpolate missing data, setting night time data to average of first and last frame
        night_start = 0
        #depth_dt['DaytimeData'] = [True if x.time() > datetime.time(8,0,0,0) and x.time() < datetime.time(1,55,0,0) else False for x in depth_dt.Time]

        daytime_data = depth_dt[depth_dt.DaytimeData == True].groupby(['RelativeDay','Trial']).agg(first_index = ('Index','first'), last_index = ('Index','last'))

        for (day,trial), (start_index,stop_index) in daytime_data.iterrows():
            stop_index = stop_index + 1
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
            #print(str(day) + '_' + str(trial))
            # Mask out data with too many nans
            non_nans = np.count_nonzero(~np.isnan(dailyData), axis = 0)
            dailyData[:,non_nans < minimumGoodData*dailyData.shape[0]] = np.nan
            #print('Nans: ' + str(np.sum(non_nans < minimumGoodData*dailyData.shape[0])))

            # Filter out data that is too close or too far from the sensor
            average_depth = np.nanmean(dailyData, axis = 0)
            median_height = np.nanmedian(dailyData)
            dailyData[:,(average_depth > median_height + max_depth) | (average_depth < median_height - max_height)] = np.nan # Filter out data 4cm lower and 8cm higher than tray
            #print('Height: ' + str(np.sum((average_depth > median_height + max_depth) | (average_depth < median_height - max_height))))


            # Smooth with savgol filter
            smoothDepthData = scipy.signal.savgol_filter(dailyData, 71, 4, axis = 0, mode = 'nearest')

        daytime_data2 = daytime_data.reset_index().groupby('RelativeDay').agg({'first_index':'min','last_index':'max'})
        # Set nighttime data as mean of before and after
        for day, (start_index,stop_index) in daytime_data2.iterrows():
            if start_index != 0: #no previous night data
                if night_start == 0:
                    depthData[night_start:start_index] = depthData[start_index]
                else:
                    depthData[night_start:start_index] = (depthData[start_index] + depthData[night_start - 1])/2
            night_start = stop_index + 1
        depthData[night_start:] = depthData[night_start-1]
        
        depth_dt['FinalGoodPixels'] = ''
        for i, frame in enumerate(depthData):
            depth_dt.loc[i,'FinalGoodPixels'] = np.sum(~np.isnan(frame))
        # Save interpolated data
        
        # Read in manual crop and mask out data outside of crop
        """with open(self.fileManager.localDepthCropFile) as f:
            for line in f:
                depth_crop_points = eval(line.rstrip())

        img = Image.new('L', (self.lp.width, self.lp.height), 0)
        ImageDraw.Draw(img).polygon(depth_crop_points, outline=1, fill=1)
        manual_crop_mask = np.array(img)
        smoothDepthData[:,manual_crop_mask == 0] = np.nan"""

        np.save(self.fileManager.localSmoothDepthFile, depthData)
        self.depth_dt = depth_dt
        depth_dt.to_csv(self.fileManager.localSmoothDepthDT)
        self.fileManager.uploadData(self.fileManager.localSmoothDepthDT)

    def createDepthFigures(self):

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
        figDaily = plt.figure(num=1, figsize=(11, rows*3 + 6))
        figDaily.suptitle(self.lp.projectID + ' Daily Depth Summary')
        gridDaily = gridspec.GridSpec(num_trials + rows +1, 1)

        current_grid_idx = 0
        hourly_dt = pd.DataFrame(columns = ['Trial_ID','Time','Volume'])
        for i in range(1,num_trials + 1):

            start_index = project_info.loc['Trial_' + str(i)].first_index
            last_index = project_info.loc['Trial_' + str(i)].last_index 
            reset_index = project_info.loc['Trial_' + str(i) + '_Reset'].last_index + 1
            #totalChangeData = vars(self.da_obj.returnVolumeSummary(self.lp.frames[start_index].time, self.lp.frames[last_index].time))

            topGrid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gridDaily[current_grid_idx])

            # Show picture of total depth change
            topAx1 = figDaily.add_subplot(topGrid[0])
            topAx1_ax = topAx1.imshow(self.da_obj.returnHeightChange(
                self.lp.frames[start_index].time, self.lp.frames[last_index].time, cropped=False), vmin=-3, vmax=3)
            bowerVolume = self.da_obj.returnVolumeSummary(self.lp.frames[start_index].time,self.lp.frames[last_index].time).depthBowerVolume
            topAx1.set_title('Total Depth Change (' + str(int(bowerVolume)) + 'cm)')
            topAx1.tick_params(colors=[0, 0, 0, 0])
            plt.colorbar(topAx1_ax, ax=topAx1)

            # Show picture of pit and castle mask
            topAx2 = figDaily.add_subplot(topGrid[1])
            topAx2_ax = topAx2.imshow(self.da_obj.returnHeightChange(self.lp.frames[reset_index].time, self.lp.frames[last_index].time, cropped = False), vmin = -3, vmax = 3)
            bowerVolume = self.da_obj.returnVolumeSummary(self.lp.frames[reset_index].time,self.lp.frames[last_index].time).depthBowerVolume
            topAx2.set_title('Reset Depth Change ('+ str(int(bowerVolume)) + 'cm)')
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

                video_start_time = max(self.lp.frames[day_start].time,self.lp.frames[day_start].time.replace(hour = 8, minute = 0, second = 0, microsecond = 0))
                video_stop_time = min(self.lp.frames[day_stop].time,self.lp.frames[day_stop].time.replace(hour = 18, minute = 0, second = 0, microsecond = 0))

                current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2]]
                current_axs[0].imshow(self.da_obj.returnHeightChange(self.lp.frames[day_info.iloc[-1].day_start].time, video_stop_time, cropped=True), vmin=-v, vmax=v)
                bowerVolume = self.da_obj.returnVolumeSummary(video_start_time, video_stop_time).depthBowerVolume
                current_axs[0].set_title(str(day) + ': ' + str(int(bowerVolume)))
                current_axs[1].imshow(self.da_obj.returnHeightChange(video_start_time, video_stop_time, cropped=True), vmin=-v, vmax=v)
                if j!=0:
                    #current_axs[2].imshow(self.da_obj.returnHeightChange(self.lp.frames[day_start].time, self.lp.frames[day_stop].time, masked=True, cropped=True), vmin=-v, vmax=v)
                    current_axs[2].imshow(self.da_obj.returnHeightChange(video_stop_time_old, video_start_time, cropped=True), vmin=-v, vmax=v)
             
                [ax.tick_params(colors=[0, 0, 0, 0]) for ax in current_axs]
                [ax.set_adjustable('box') for ax in current_axs]

                #if day == 7:
                #    self.da_obj.returnHeightChange(self.lp.frames[day_info.iloc[-1].day_start].time, self.lp.frames[day_stop].time, cropped = True, pdb_flag=True)

                good_data_start = self.lp.frames[day_start].time
                good_data_stop = self.lp.frames[day_stop].time
                day_stamp = self.lp.frames[day_start].time.replace(hour = 0, minute=0, second=0, microsecond=0)


                for k in range(8,18):
                    start = max(day_stamp + datetime.timedelta(hours=k), good_data_start)
                    if k == 8:
                        try:
                            volume = self.da_obj.returnVolumeSummary(previous_stop, start).depthBowerVolume
                            hourly_dt.loc[len(hourly_dt.index)] = ['Trial_' + str(i), start - datetime.timedelta(hours = 12)/2,volume]
                        except NameError:
                            pass

                    stop = min(day_stamp + datetime.timedelta(hours=k+1), good_data_stop)
                    volume = self.da_obj.returnVolumeSummary(max(start,good_data_start),min(stop,good_data_stop)).depthBowerVolume
                    hourly_dt.loc[len(hourly_dt.index)] = ['Trial_' + str(i),start.replace(minute = 30),volume]

                    previous_stop = stop

                video_stop_time_old = video_stop_time


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



