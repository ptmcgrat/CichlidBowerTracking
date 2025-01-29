import argparse, datetime, gspread, time, pdb, warnings, psutil
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
from oauth2client.service_account import ServiceAccountCredentials
import oauth2client
from skimage import morphology
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('Logfile', type = str, help = 'Name of logfile')
args = parser.parse_args()


class DriveUpdater:
    def __init__(self, logfile):
        self.lp = LP(logfile)

        self.fileManager = FM(projectID = self.lp.projectID, analysisID = self.lp.analysisID)
        self.node = self.lp.uname.split("node='")[1].split("'")[0]
        self.lastFrameTime = self.lp.frames[-1].time
        self.masterDirectory = self.fileManager.localMasterDir
        self.projectDirectory = self.fileManager.localProjectDir
        
        self.googleController = GC(self.fileManager.localCredentialSpreadsheet)
        self.googleController.addProjectID(self.lp.projectID, self.fileManager.localProjectDir + 'GoogleErrors.txt')

        self._createImage()

        self.credentialDrive = self.fileManager.localCredentialDrive

        self._uploadImage(self.projectDirectory + self.lp.tankID + '.jpg', self.projectDirectory + self.lp.tankID + '_2.jpg', self.lp.tankID, self.lp.tankID + '_2.jpg')
    
    def _filterPixels(self, pixels):
            """
            try:
                self.badPixels
            except AttributeError:
                # Identify bad pixels
                std_template = np.load(self.projectDirectory + daylightFrames[0].std_file)
                stds = np.zeros(shape = (min(len(days), 10), std_template.shape[0], std_template.shape[1]), dtype = std_template.dtype())
                # Read in std deviation data to determine threshold
                #moved
                dpth_dif = np.load(self.projectDirectory + self.lp.frames[-1].npy_file)
                median_height = np.nanmedian(dpth_3)
            """
            return pixels
        
    def _calculateBower(self, depthChange, th):
        """
        daily_bower = daily_change.copy()
        thresholded_change = np.where((daily_change >= 0.4) | (daily_change <= -0.4), True, False)
        thresholded_change = morphology.remove_small_objects(thresholded_change,1000).astype(int)
        daily_bower[(thresholded_change == 0) & (~np.isnan(daily_change))] = 0
        """
        daily_bower = depthChange.copy()
        thresholded_change = np.where((depthChange >= th) | (depthChange <= -th), True, False)
        thresholded_change = morphology.remove_small_objects(thresholded_change,1000).astype(int)
        daily_bower[(thresholded_change == 0) & (~np.isnan(depthChange))] = 0
        return daily_bower
    
    def _createImage(self, stdcutoff = 0.1):
        if len(self.lp.frames) > 1 and self.lp.frames[-1].std< 0.00001 and self.lp.frames[-1].gp==self.lp.frames[-2].gp:
            self.googleController.modifyPiGS('DataDuplicated', 'Yes')
        else: 
            self.googleController.modifyPiGS('DataDuplicated', 'No')


        lastHourFrames = [x for x in self.lp.frames if x.time > self.lastFrameTime - datetime.timedelta(hours = 1)] # frames from the last hour
        lastTwoHourFrames = [x for x in self.lp.frames if x.time > self.lastFrameTime - datetime.timedelta(hours = 2)] # frames from the last two hours
        daylightFrames = [x for x in self.lp.frames if x.time.hour >= 8 and x.time.hour <= 17] # frames during daylight

        th_change = str(self.lastFrameTime-lastTwoHourFrames[0].time)
        h_change = str(self.lastFrameTime - lastHourFrames[0].time)

        # Dictionary to hold all the unique days that have daylight frames
        days={}
        
        for x in daylightFrames:
            days.update({str(x.time.day)+' '+str(x.time.month):x.time.month})
            # This way we only identify days that have frames during the daylight

        # Determine the size of the figure and create it
        num_rows = 3 + len(days) # First pic rows, 1 hour, 2 hour, then 1 row for each unique day
        axes = [0]*3*num_rows # Hold axes in lis
        
        fig = plt.figure(figsize=(14,4*num_rows + 1))
        fig.suptitle(self.lp.projectID + ' ' + str(self.lastFrameTime))
        
        plt.rcParams.update({'font.size': 18})
        
        # Create subplots
        for i in range(num_rows):
            #pdb.set_trace()
            axes[3*i + 0] = fig.add_subplot(num_rows, 3, 3*i + 1) # Filtered Absolute Depth
            axes[3*i + 1] = fig.add_subplot(num_rows, 3, 3*i + 2) # Relative Depth Change
            axes[3*i + 2] = fig.add_subplot(num_rows, 3, 3*i + 3) # Bower changes
       
        # Set titles of each plot, strating with the first three rows
        axes[0].set_title('Kinect RGB Picture')
        axes[1].set_title('PiCamera RGB Picture')
        axes[2].set_title('Total Depth Change' )
        axes[3].set_title('Hour ago Depth')
        axes[4].set_title('Last hour change\n'+h_change)
        axes[5].set_title('Last hour bower\n')
        axes[6].set_title('2 Hour ago Depth')
        axes[7].set_title('Last 2 hours change\n'+th_change)
        axes[8].set_title('Last 2 hours bower\n')
        
        #pdb.set_trace()
        # Now set titles of the unknown number of days
        for i, day in enumerate([x for x in days.keys()][::-1]):
            axes[3*i+9].set_title(str(days[day]) + '/' + str(day) + ' Depth Data')
            axes[3*i+10].set_title(str(days[day]) + '/' + str(day) + ' Daytime Depth Change')
            axes[3*i+11].set_title(str(days[day]) + '/' + str(day) + ' Identified Bower')
        
        # Plot data  for first two rows
        img_1 = img.imread(self.projectDirectory + self.lp.frames[-1].pic_file)
        try:
            img_2 = img.imread(self.projectDirectory + self.lp.movies[-1].pic_file)
        except:
            img_2 = img_1
        
        depth_last = self._filterPixels(np.load(self.projectDirectory + self.lp.frames[-1].npy_file))
        depth_first = self._filterPixels(np.load(self.projectDirectory + daylightFrames[0].npy_file))
        depth_hour = self._filterPixels(np.load(self.projectDirectory + lastHourFrames[0].npy_file))
        depth_twohours = self._filterPixels(np.load(self.projectDirectory + lastTwoHourFrames[0].npy_file))

        median_height = np.nanmedian(depth_first)

        axes[0].imshow(img_1)
        axes[1].imshow(img_2)
        axes[2].imshow(depth_last - depth_first, vmin = -2, vmax = 2)
        axes[3].imshow(depth_hour, vmin = median_height - 8, vmax = median_height + 8)
        axes[4].imshow(depth_last - depth_hour, vmin = -0.5, vmax = 0.5)
        axes[5].imshow(self._calculateBower(depth_last - depth_hour, 0.2), vmin = -1, vmax = 1)
        axes[6].imshow(depth_twohours, vmin = median_height - 8, vmax = median_height + 8)
        axes[7].imshow(depth_last - depth_twohours, vmin = -0.5, vmax = 0.5)
        axes[8].imshow(self._calculateBower(depth_last - depth_twohours, 0.2), vmin = -0.5, vmax = 0.5)

        for i,date in enumerate([x for x in days.keys()][::-1]):
            day=date.split(' ')[0]
            month=date.split(' ')[1]
            daylightFrames_month = [x for x in daylightFrames if x.time.month == int(month) ]
            daylightFrames_day = [x for x in daylightFrames_month if x.time.day == int(day) ]
            
            if daylightFrames_day==[]:
                print(date)
                # frames during daylight
            depth_start = self._filterPixels(np.load(self.projectDirectory + daylightFrames_day[0].npy_file))
            depth_stop = self._filterPixels(np.load(self.projectDirectory + daylightFrames_day[-1].npy_file))

            axes[3*i+9].imshow(depth_start, vmin = median_height - 8, vmax = median_height + 8)
            axes[3*i+10].imshow(depth_stop - depth_start, vmin = -1, vmax = 1)
            axes[3*i+11].imshow(self._calculateBower(depth_stop - depth_start, 0.4), vmin = -1, vmax = 1)

        #plt.subplots_adjust(bottom = 0.15, left = 0.12, wspace = 0.24, hspace = 0.57)
        plt.savefig(self.projectDirectory + self.lp.tankID + '.jpg')
        #return self.graph_summary_fname

        fig = plt.figure(figsize=(6,3))
        fig.tight_layout()
        ax1 = fig.add_subplot(1, 2, 1) #Pic from Kinect
        ax2 = fig.add_subplot(1, 2, 2) #Pic from Camera
        ax1.imshow(depth_last - depth_twohours, vmin = -.75, vmax = .75)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        ax2.imshow(depth_last - depth_hour, vmin = -.75, vmax = .75) # +- 1 cms
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

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
