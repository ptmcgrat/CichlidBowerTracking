from helper_modules.depth_analyzer import DepthAnalyzer as DA
from helper_modules.depth_analyzer import ClusterAnalyzer as CA
from helper_modules.file_manager import FileManager as FM
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import (cm, colors, gridspec, ticker)
import numpy as np
from helper_modules.log_parser import LogParser as LP
import pdb
import math
import datetime

class SummaryPrep:

    def __init__(self,analysisID = 'MC_multi',projectID = 'MC_s9_tr6_BowerBuilding'):
        
        self.fileManager = FM(analysisID=analysisID,projectID=projectID)
        self.da_obj = DA(self.fileManager)
        self.ca_obj = CA(self.fileManager)
        # self.lp_obj = LP()
        self.downloadProjectData()
        self.createSummary()


        
    def downloadProjectData(self):

        # self.fileManager.createDirectory(self.fileManager.localMasterDir)
        # self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
        # self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
        # self.fileManager.createDirectory(self.fileManager.localTempDir)
        # # self.fileManager.createDirectory(self.fileManager.localAllClipsDir)
        # self.fileManager.createDirectory(self.fileManager.localManualLabelClipsDir)
        # self.fileManager.createDirectory(self.fileManager.localManualLabelFramesDir)
        # self.fileManager.createDirectory(self.fileManager.localLogfileDir)

        self.fileManager.downloadData(self.fileManager.localLogfile)
        # self.fileManager.downloadData(self.videoObj.localVideoFile)
        self.fileManager.createDirectory(self.fileManager.localSummaryDir)
        self.fileManager.downloadData(self.fileManager.localFrameDir, tarred = True)
        # self.fileManager.downloadData(self.fileManager.localDepthCropFile)

        pass

    def uploadProjectData(self,delete=False):
        self.fileManager.uploadData(self.fileManager.localSummaryDir)
    
    def createSummary(self):

        self.depth_dt = pd.read_csv(self.fileManager.localSmoothDepthDT, index_col = 0)
        self.depth_dt = self.depth_dt[~self.depth_dt.Trial.isna()]

        project_info = self.depth_dt[self.depth_dt.DaytimeData == True].groupby('Trial').agg(first_index = ('Index','first'), last_index = ('Index','last'))
        num_trials = len(self.fileManager.lp.tankresetstart)
        rows = int(np.ceil((self.depth_dt[~(self.depth_dt.Trial.str.contains('Reset')) & ~(self.depth_dt.Trial == '')].groupby(['Trial']).nunique()['RelativeDay']/10)).sum())



        # unique_videos = cluster_dt_filtered['VideoID'].unique()
        # legend_labels = { "c": "Build Scoop", "s": "Quiver/Spawn", "p": "Build Spit", "b": "Build Multiple","f": "Feed Scoop", "t": "Feed Spit", "m": "Feed Multiple" }

        self.TransM = np.load(self.fileManager.localTransMFile)
        self.cluster_dt = pd.read_csv(self.fileManager.localAllLabeledClustersFile,index_col = 'TimeStamp',parse_dates=True)
        # cluster_dt = pd.read_csv(self.fileManager.localAllLabeledClustersFile)
        required_columns = {'VideoID', 'X_depth', 'Y_depth', 'predicted_label'}
        behaviors_of_interest = ['c','p','f','t','b','m','s']
        cluster_dt_filtered = self.cluster_dt[self.cluster_dt['predicted_label'].isin(behaviors_of_interest)]

        colors = plt.cm.get_cmap('tab10', len(behaviors_of_interest))
        color_map = {behavior: colors(i) for i, behavior in enumerate(behaviors_of_interest)}
        legend_labels = { "c": "Build Scoop", "s": "Quiver/Spawn", "p": "Build Spit", "b": "Build Multiple","f": "Feed Scoop", "t": "Feed Spit", "m": "Feed Multiple" }
        # legend_labels = { "c": "Build Scoop",  "p": "Build Spit", "f": "Feed Scoop", "t": "Feed Spit","b":"Build Multiple","m":"Feed Multiple"}
        
        # pdb.set_trace()
        
        # figDaily, axes = plt.subplots(num_trials+rows+2,1,figsize=(15,rows*5+12))
        # pdb.set_trace()
        # # figDaily = plt.figure()
        # figDaily = plt.figure(num=1, figsize=(15, rows*4 + 12))
        # figDaily.suptitle(self.fileManager.lp.projectID + ' Daily Depth Summary')
        # gridDaily = gridspec.GridSpec(num_trials +rows+1, 1)
        # current_grid_idx = 0
        # pdb.set_trace()

        for i in range(1,num_trials + 1):

            start_index = project_info.loc['Trial_' + str(i)].first_index
            last_index = project_info.loc['Trial_' + str(i)].last_index
            reset_index = project_info.loc['Trial_' + str(i) + '_Reset'].last_index

            #totalChangeData = vars(self.da_obj.returnVolumeSummary(self.lp.frames[start_index].time, self.lp.frames[last_index].time))

            # topGrid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gridDaily[current_grid_idx])

            # Show picture of total depth change
            # topAx1 = figDaily.add_subplot(topGrid[0])
            # topAx1_ax = topAx1.imshow(self.da_obj.returnHeightChange(
            #     self.fileManager.lp.frames[start_index].time, self.fileManager.lp.frames[last_index].time, cropped=False), vmin=-3, vmax=3)
            # bowerVolume = self.da_obj.returnVolumeSummary(self.fileManager.lp.frames[start_index].time,self.fileManager.lp.frames[last_index].time).depthBowerVolume
            # topAx1.set_title('Total Depth Change (' + str(int(bowerVolume)) + 'cm)')
            # topAx1.tick_params(colors=[0, 0, 0, 0])
            # plt.colorbar(topAx1_ax, ax=topAx1)

            # # Show picture of pit and castle mask
            # topAx2 = figDaily.add_subplot(topGrid[1])
            # topAx2_ax = topAx2.imshow(self.da_obj.returnHeightChange(self.fileManager.lp.frames[reset_index].time, self.fileManager.lp.frames[last_index].time, cropped = False), vmin = -3, vmax = 3)
            # bowerVolume = self.da_obj.returnVolumeSummary(self.fileManager.lp.frames[reset_index].time,self.fileManager.lp.frames[last_index].time).depthBowerVolume
            # topAx2.set_title('Reset Depth Change ('+ str(int(bowerVolume)) + 'cm)')
            # topAx2.tick_params(colors=[0, 0, 0, 0])
            # plt.colorbar(topAx2_ax, ax=topAx2)

            day_info = self.depth_dt[(self.depth_dt.DaytimeData == True)&(self.depth_dt.Trial == 'Trial_' + str(i))].groupby('RelativeDay').agg(day_start = ('Index','first'), day_stop = ('Index','last')).sort_index(ascending = False)

            # num_days = min(len(day_info),10)
            num_days = len(day_info)
            v = 2.0
            nrows= len(behaviors_of_interest)+3
            figDaily, axes = plt.subplots(nrows,num_days,figsize=(num_days*5,nrows*4))
            figDaily.suptitle("Combined Summary: " + self.fileManager.projectID+"Trial"+str(i)+"\n\n", fontsize=16)  
            # pdb.set_trace()

            for j, (day,(day_start,day_stop)) in enumerate(day_info.iterrows()):
                # pdb.set_trace()
                # if j % num_days == 0:
                #     if j!=0:
                #         cax = figDaily.add_subplot(midGrid[:, -1])
                #         plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)

                #     # current_grid_idx += 1
                #     # midGrid = gridspec.GridSpecFromSubplotSpec(4, num_days + 1, subplot_spec=gridDaily[current_grid_idx],hspace =0.4,wspace = 0.3)

                # current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2, 3]]
                # # current_axs = [figDaily.add_subplot(midGrid[n, (num_days - j % num_days) - 1]) for n in [0, 1, 2, 3]]
                # if j % num_days == 0 and j!=0:
                # pdb.set_t/race()
                axes[0,num_days-j-1].imshow(self.da_obj.returnHeightChange(self.fileManager.lp.frames[day_info.iloc[-1].day_start].time, self.fileManager.lp.frames[day_stop + 1].time, cropped=True), vmin=-v, vmax=v)
                # pdb.set_trace()
                bowerVolume = self.da_obj.returnVolumeSummary(self.fileManager.lp.frames[day_start-1].time,self.fileManager.lp.frames[day_stop+1].time).depthBowerVolume
                axes[0,num_days-j-1].set_title(str(day) + ': ' + str(int(bowerVolume)))
                axes[1,num_days-j-1].imshow(self.da_obj.returnHeightChange(self.fileManager.lp.frames[day_start-1].time, self.fileManager.lp.frames[day_stop+1].time, cropped=True), vmin=-v, vmax=v)
                axes[2,num_days-j-1].imshow(self.da_obj.returnHeightChange(self.fileManager.lp.frames[day_start-1].time, self.fileManager.lp.frames[day_stop+1].time, masked=True, cropped=True), vmin=-v, vmax=v)
                depth_ref = self.da_obj.returnHeightChange(self.fileManager.lp.frames[day_start-1].time, self.fileManager.lp.frames[day_stop+1].time, masked=True, cropped=True)
                [ax.tick_params(colors=[0, 0, 0, 0]) for ax in axes.flatten()]
                [ax.set_adjustable('box') for ax in axes.flatten()]
                height, width = depth_ref.shape

                # good_data_start = self.fileManager.lp.frames[day_start].time
                # good_data_stop = self.fileManager.lp.frames[day_stop].time
                # day_stamp = self.fileManager.lp.frames[day_start].time.replace(hour = 0, minute=0, second=0, microsecond=0)
                # pdb.set_trace()
                clusters_today = cluster_dt_filtered[(cluster_dt_filtered.index >= self.fileManager.lp.frames[day_start].time) & 
                                (cluster_dt_filtered.index <= self.fileManager.lp.frames[day_stop].time)]

                axes[3,num_days-j-1].set_xlim([0, width])
                axes[3,num_days-j-1].set_ylim([0, height]) 

                # Plot the clusters as scatter plot in the fourth row
                # current_axs[3].scatter(clusters_today['X_depth'], clusters_today['Y_depth'], c='red', alpha=0.6, marker='o', s=10)
                k=3
                for behavior in behaviors_of_interest:
                    subset = clusters_today[clusters_today['predicted_label'] == behavior]
                    axes[k,num_days-j-1].scatter(
                        subset['Y_depth'], subset['X_depth'],
                        label=legend_labels.get(behavior, behavior),
                        color=color_map.get(behavior, 'black'),
                        alpha=0.7,
                        s=10
                    )
                    k+=1
                
                axes[3,num_days-j-1].set_title(f'Clusters - Day {day}',fontsize =8)
                for k in range(3,3+len(behaviors_of_interest)):
                    axes[k,num_days-j-1].legend(fontsize='x-small') 
                # current_axs[3].invert_xaxis()
                # for k in range(8,20):
                #     start = day_stamp + datetime.timedelta(hours=k)
                #     stop = day_stamp + datetime.timedelta(hours=k+1)
                #     if stop < good_data_start or start > good_data_stop:
                #         continue
                #     volume = self.da_obj.returnVolumeSummary(max(start,good_data_start),min(stop,good_data_stop)).depthBowerVolume
                    # hourly_dt.loc[len(hourly_dt.index)] = ['Trial_' + str(i),start.replace(minute = 30),volume]


            # cax = figDaily.add_subplot(midGrid[:, -1])
            # plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-v, vmax=v), cmap='viridis'), cax=cax)
            # current_grid_idx += 1 
            plt.tight_layout()
            # plt.show()
            plt.savefig("/home/pkolipaka3@ad.gatech.edu/Temp/CichlidAnalyzer/__ProjectData/MC_multi/MC_s9_tr6_BowerBuilding/Summary/CombinedSummary"+str(i)+".pdf")
            plt.savefig(self.fileManager.localCombinedSummaryFigure)

sa_obj = SummaryPrep()
