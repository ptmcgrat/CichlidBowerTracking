
import argparse, datetime, gspread, time, pdb, warnings, psutil, shutil, os
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
from cichlid_bower_tracking.helper_modules.log_parser import LogParser as LP
from cichlid_bower_tracking.helper_modules.googleController import GoogleController as GC

import matplotlib
matplotlib.use('Pdf')  # Enables creation of pdf without needing to worry about X11 forwarding when ssh'ing into the Pi
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import oauth2client
from PIL import Image 

parser = argparse.ArgumentParser()
parser.add_argument('Logfile', type = str, help = 'Name of logfile')
args = parser.parse_args()


class DriveUpdater:
    def __init__(self, logfile):
        self.lp = LP(logfile)

        self.fileManager = FM(projectID = self.lp.projectID, analysisID = self.lp.analysisID)
        self.lastFrameTime = self.lp.frames[-1].time
        self.projectDirectory = self.fileManager.localProjectDir
        
        self.googleController = GC(self.fileManager.localCredentialSpreadsheet)
        self.googleController.addProjectID(self.lp.projectID, self.fileManager.localProjectDir + 'GoogleErrors.txt')

        self._createImage()

        self.credentialDrive = self.fileManager.localCredentialDrive

        self._uploadImage(self.projectDirectory + self.lp.tankID + '.jpg', self.projectDirectory + self.lp.tankID + '_2.jpg', self.lp.tankID, self.lp.tankID + '_2.jpg')
    
    def _filterPixels(self, pixels):
            
            try:
                self.depth_max
            except AttributeError:
                mask_file = self.fileManager.localMasterDir + "__TankData/"+ self.lp.tankID + "/MaskedImg.jpg"
                if self.fileManager.checkFileExists(mask_file):
                    self.fileManager.downloadData(mask_file)
                    mask = Image.open(mask_file)
                    mask = mask.convert("L")
                    mask = np.array(mask)
                    self.depth_mask = mask != 0
                else:
                    raw_file = self.fileManager.localMasterDir + "__TankData/"+ self.lp.tankID + "/FirstDepthRGB.jpg"
                    if not self.fileManager.checkFileExists(raw_file):
                        os.makedirs(os.path.dirname(raw_file), exist_ok=True)
                        shutil.copyfile(self.fileManager.localFirstDepthRGB, raw_file)
                        self.fileManager.uploadData(raw_file)
                    self.depth_mask = np.ones(shape = (480,640))
            
            pixels[self.depth_mask == False] = np.nan
            
            return pixels
        
    def _calculateBower(self, daily_bower, bower_mask):
        """
        daily_bower = daily_change.copy()
        thresholded_change = np.where((daily_change >= 0.4) | (daily_change <= -0.4), True, False)
        thresholded_change = morphology.remove_small_objects(thresholded_change,1000).astype(int)
        daily_bower[(thresholded_change == 0) & (~np.isnan(daily_change))] = 0
        """
        # daily_bower = depthChange.copy()
        # thresholded_change = np.where((depthChange >= th) | (depthChange <= -th), True, False)
        # thresholded_change = morphology.remove_small_objects(thresholded_change,1000).astype(int)
        # daily_bower[(thresholded_change == 0) & (~np.isnan(depthChange))] = 0
        # return daily_bower
        # thresholds = [0.2,0.3,0.5,0.7,0.9, 1.2, 1.3, 1.5, 1.7, 2]
        #daily_bower = depthChange.copy()

        #bower_mask = np.where(total_depth_change < (-1*threshold), -1, np.where(total_depth_change > threshold,1,0))
        volume_pit = np.nansum(daily_bower[np.where(bower_mask == 1)])* self.fileManager.pixelLength ** 2
        volume_castle = np.nansum(daily_bower[np.where(bower_mask == -1)])* self.fileManager.pixelLength ** 2
        # print("volume_castle", volume_castle)
        # print("volume_pit",volume_pit)
        # pdb.set_trace()
        # daily_bower [(mask == 0)] = 0
        # daily_bower[(self.depth_mask == 0)] = np.nan

        #bower_mask = np.where(total_depth_change < -1.0, -1, np.where(total_depth_change > 1.0,1,0))
        #daily_bower [(bower_mask == 0)] = 0
        #daily_bower[(self.depth_mask == 0)] = np.nan

        return volume_pit, volume_castle
    
    def _createImage(self, stdcutoff = 0.1):
        # Creates an image to describe the previous round of building. Current setup is:
        # 1st row: Depth Sensory RGB; PiCamera RGB; Current Depth; First Depth; Current Day of Building
        # For each trial
        # 2nd row: First pic; last pic; Total change; Reset change (if available); daily volume info

        # Check to see if the data is duplicated
        if len(self.lp.frames) > 1 and self.lp.frames[-1].std < 0.00001 and self.lp.frames[-1].gp==self.lp.frames[-2].gp:
            self.googleController.modifyPiGS('DataDuplicated', 'Yes')
        else: 
            self.googleController.modifyPiGS('DataDuplicated', 'No')

        # Calulate how many trials
        if self.lp.tankresetstop:
            num_trials = len(self.lp.tankresetstop) + 1
        else:
            num_trials = 1

        # Determine the size of the figure and create it
        num_rows = num_trials + 1 # First row is general, rest of rows are per trial   
        fig = plt.figure(figsize=(20,4*num_rows + 1))
        fig.suptitle(self.lp.projectID + ' ' + str(self.lastFrameTime), fontsize=24)
        #plt.rcParams.update({'font.size': 18})
        axes = []

        # Grab daylight frames

        
        # Create first row
        daylightFrames = [x for x in self.lp.frames if x.time.hour >= 8 and x.time.hour <= 17] # frames during daylight        
        daylightFrames_day = [x for x in daylightFrames if x.time.day == daylightFrames[-1].time.day and x.time.month == daylightFrames[-1].time.month]

        for i in range(5):
            axes.append(fig.add_subplot(num_rows, 5, i+1))
        axes[0].set_title('Latest Depth RGB')
        axes[1].set_title('Latest PiCamera RGB')
        axes[2].set_title('First Depth')
        axes[3].set_title('Current Depth')
        axes[4].set_title('Last day change')

        #print('Starting' + str(datetime.datetime.now()))
        img_1 = img.imread(self.projectDirectory + self.lp.frames[-1].pic_file)
        img_2 = img.imread(self.projectDirectory + self.lp.movies[-1].pic_file)
        #print('Current images read' + str(datetime.datetime.now()))
        depth_first = self._filterPixels(np.load(self.projectDirectory + daylightFrames[0].npy_file))
        #print('Depth read' + str(datetime.datetime.now()))
        depth_last = self._filterPixels(np.load(self.projectDirectory + self.lp.frames[-1].npy_file))
        #print('Depth 2 read' + str(datetime.datetime.now()))
        depth_dayend = self._filterPixels(np.load(self.projectDirectory + daylightFrames[-1].npy_file))
        #print('Depth 3 read' + str(datetime.datetime.now()))
        depth_daystart = self._filterPixels(np.load(self.projectDirectory + daylightFrames_day[0].npy_file))
        #print('Depth 4 read' + str(datetime.datetime.now()))
        
        median_height = np.nanmedian(depth_first)

        axes[0].imshow(img_1)
        axes[1].imshow(img_2)
        axes[2].imshow(depth_first, vmin = median_height - 4, vmax = median_height + 4)
        axes[3].imshow(depth_last, vmin = median_height - 4, vmax = median_height + 4)
        axes[4].imshow(depth_dayend - depth_daystart, vmin = -2, vmax = 2)

        for i in range(5):
            axes[i].set_xticks([])
            axes[i].set_yticks([])


        for j in range(num_trials):
            for i in range(5):
                axes.append(fig.add_subplot(num_rows, 5, 5*(j+1) + i+1))

        for j in range(num_trials):
            if not self.lp.tankresetstop:
                trial_frames = [x for x in daylightFrames]
                trial_movies = [x for x in self.lp.movies]
            elif j == 0:
                trial_frames = [x for x in daylightFrames if x.time < self.lp.tankresetstart[j]]
                trial_movies = [x for x in self.lp.movies if x.startTime < self.lp.tankresetstart[j]]

            elif j == num_trials - 1:
                trial_frames = [x for x in daylightFrames if x.time > self.lp.tankresetstop[j-1]]
                trial_movies = [x for x in self.lp.movies if x.startTime > self.lp.tankresetstop[j-1]]
            else:
                trial_frames = [x for x in daylightFrames if x.startTime > self.lp.tankresetstop[j-1] and x.startTime < self.lp.tankresetstart[j]]
                trial_movies = [x for x in daylightFrames if x.startTime > self.lp.tankresetstop[j-1] and x.startTime < self.lp.tankresetstart[j]]

            days = {}
            for x in trial_frames:
                days[x.time.day] = 1
            print(str(j) + ':' + str(days)) 

            img_1 = img.imread(self.projectDirectory + trial_frames[-1].pic_file)
            img_2 = img.imread(self.projectDirectory + trial_movies[-1].pic_file)
            
            depth_first = self._filterPixels(np.load(self.projectDirectory + trial_frames[0].npy_file))
            depth_last = self._filterPixels(np.load(self.projectDirectory + trial_frames[-1].npy_file))
            
            castle = []
            pit = []
            total_depth_change = depth_last - depth_first
            threshold = 1
            bower_mask = np.where(total_depth_change < (-1*threshold), -1, np.where(total_depth_change > threshold,1,0))
            for current_day in days:
                #print('Volume calculated' + str(datetime.datetime.now()))
                day_data = [x for x in trial_frames if x.time.day == current_day]
                day_start = self._filterPixels(np.load(self.projectDirectory + day_data[0].npy_file))
                day_stop = self._filterPixels(np.load(self.projectDirectory + day_data[-1].npy_file))
                day_change = day_stop - day_start
                pit_day, castle_day = self._calculateBower(day_change, bower_mask)
                pit.append(pit_day)
                castle.append(castle_day)


            if j != num_trials - 1:
                try:
                    reset_depth_frame = [x for x in daylightFrames if x.time > self.lp.tankresetstop[j]][0]
                except IndexError:
                    pdb.set_trace()
                reset_depth = self._filterPixels(np.load(self.projectDirectory + reset_depth_frame.npy_file))
            offset = (num_trials - j) * 5
            axes[offset].imshow(img_1)
            axes[offset].set_ylabel('Trial ' + str(j+1),fontsize=16, rotation=90, labelpad=20)# ha ='right')

            axes[offset + 1].imshow(img_2)

            axes[offset + 2].imshow(depth_last-depth_first, vmin = -2, vmax = 2)
            if j != num_trials - 1:
                axes[offset + 3].imshow(depth_last - reset_depth, vmin = -2, vmax = 2)
            plotdays = [x + 1 for x in range(len(days))]
            axes[offset + 4].plot(plotdays, pit, '-o', color = 'blue', label = 'Pit volume', alpha = 0.7)
            axes[offset + 4].plot(plotdays, castle, '-o', color = 'red', label = 'Castle volume', alpha = 0.7)
            axes[offset + 4].set_ylim(-500,500)
            for i in range(4):
                axes[offset + i].set_xticks([])
                axes[offset + i].set_yticks([])


        #plt.subplots_adjust(bottom = 0.15, left = 0.12, wspace = 0.24, hspace = 0.57)
        fig.subplots_adjust(left=0.2, hspace=0.4)
        plt.savefig(self.projectDirectory + self.lp.tankID + '.jpg')
        #return self.graph_summary_fname

        fig = plt.figure(figsize=(3,3))
        fig.tight_layout()
        fig.subplots_adjust(left=0.2, hspace=0.4)
        ax1 = fig.add_subplot(1, 1, 1) #Pic from Kinect
        #ax2 = fig.add_subplot(1, 2, 2) #Pic from Camera

        ax1.imshow(depth_dayend - depth_daystart, vmin = -1, vmax = 1)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        #ax2.imshow(depth_last - depth_hour, vmin = -.75, vmax = .75) # +- 1 cms
        #ax2.axes.get_xaxis().set_visible(False)
        #ax2.axes.get_yaxis().set_visible(False)

        ax1.set_xticklabels([])  # Hide x-axis labels
        ax1.set_yticklabels([]) 
        #ax2.set_xticklabels([])  # Hide x-axis labels
        #ax2.set_yticklabels([]) 

        fig.tight_layout()

        fig.savefig(self.projectDirectory + self.lp.tankID + '_2.jpg')

        #Update PiStatus
        current_temp = psutil.sensors_temperatures()['cpu_thermal'][0][1]
        harddrive_use = psutil.disk_usage(self.fileManager.localMasterDir)[3]
        cpu_use = psutil.cpu_percent()
        ram_use = psutil.virtual_memory()[2]

        self.googleController.modifyPiGS('PiStatus','Temperature: ' + str(current_temp) + ',,HardDriveUsage: ' + str(harddrive_use) + ',,CPUUsage: ' + str(cpu_use) + ',,RAMUse: ' + str(ram_use))

    
    def _uploadImage(self, image_file1, image_file2, name1, name2): #name should have format 't###_icon' or 't###_link'
        self._authenticateGoogleDrive()
        drive = GoogleDrive(self.gauth)
        folder_id = "'151cke-0p-Kx-QjJbU45huK31YfiUs6po'"  #'Public Images' folder ID
        
        try:
            file_list = drive.ListFile({'q':"{} in parents and trashed=false".format(folder_id)}).GetList()
        except oauth2client.clientsecrets.InvalidClientSecretsError:
            self._authenticateGoogleDrive()
            file_list = drive.ListFile({'q':"{} in parents and trashed=false".format(folder_id)}).GetList()
        #print(file_list)
        # check if file name already exists so we can replace it
        flag1 = False
        flag2 = False
        count = 0
        while flag1 == False and count < len(file_list):
            if file_list[count]['title'] == name1:
                fileID1 = file_list[count]['id']
                flag1 = True
            else:
                count += 1
        count = 0
        while flag2 == False and count < len(file_list):
            if file_list[count]['title'] == name2:
                fileID2 = file_list[count]['id']
                flag2 = True
            else:
                count += 1

        if flag1 == True:
            # Replace the file if name exists
            f1 = drive.CreateFile({'id': fileID1})
            f1.SetContentFile(image_file1)
            f1.Upload()
            # print("Replaced", name, "with newest version")
        else:
            # Upload the image normally if name does not exist
            f1 = drive.CreateFile({'title': name1, 'mimeType':'image/jpeg',
                                 "parents": [{"kind": "drive#fileLink", "id": folder_id[1:-1]}]})
            f1.SetContentFile(image_file1)
            f1.Upload()                   
            # print("Uploaded", name, "as new file")

        if flag2 == True:
            # Replace the file if name exists
            f2 = drive.CreateFile({'id': fileID2})
            f2.SetContentFile(image_file2)
            f2.Upload()
            # print("Replaced", name, "with newest version")
        else:
            # Upload the image normally if name does not exist
            f2 = drive.CreateFile({'title': name2, 'mimeType':'image/jpeg',
                                 "parents": [{"kind": "drive#fileLink", "id": folder_id[1:-1]}]})
            f2.SetContentFile(image_file2)
            f2.Upload()                   
            # print("Uploaded", name, "as new file")


        info = '=HYPERLINK("' + f1['webContentLink'].replace('&export=download', '') + '", IMAGE("' + f2['webContentLink'] + '"))'

        #info = '=HYPERLINK("' + f['alternateLink'] + '", IMAGE("' + f['webContentLink'] + '"))'
        self.googleController.modifyPiGS('Image', info, ping = False)
    
    def _authenticateGoogleDrive(self):
        self.gauth = GoogleAuth()
        # Try to load saved client credentials
        self.gauth.LoadCredentialsFile(self.credentialDrive)
        if self.gauth.credentials is None:
            # Authenticate if they're not there
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh them if token is expired
            self.gauth.Refresh()
        else:
            # Initialize with the saved creds
            self.gauth.Authorize()
        # Save the current credentials to a file
        self.gauth.SaveCredentialsFile(self.credentialDrive)

dr_obj = DriveUpdater(args.Logfile)
#try:
#    dr_obj = DriveUpdater(args.Logfile)
#except Exception as e:
#    print(f'skipping drive update due to error: {e}')
