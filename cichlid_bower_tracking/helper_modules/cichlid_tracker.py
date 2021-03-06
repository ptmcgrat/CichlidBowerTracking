import platform, sys, os, shutil, datetime, subprocess, gspread, time, socket, pdb, time
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
from cichlid_bower_tracking.helper_modules.log_parser import LogParser as LP

from picamera import PiCamera
import numpy as np

import warnings
warnings.filterwarnings('ignore')


#with warnings.catch_warnings():
#    warnings.filterwarnings('ignore', message = 'Degrees of freedom <= 0 for slice.')
#    warnings.filterwarnings('ignore', message = 'Mean of empty slice')
    
from PIL import Image
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.image

class CichlidTracker:
    def __init__(self):
        

        # 1: Define valid commands and ignore warnings
        self.commands = ['New', 'Restart', 'Stop', 'Rewrite', 'UploadData', 'LocalDelete', 'Snapshots']
        np.seterr(invalid='ignore')

        # 2: Determine which Kinect is attached (This script can handle v1 or v2 Kinects)
        self._identifyDevice() #Stored in self.device
        self.system = platform.node()

        # 3: Create file manager
        self.fileManager = FM()

        # 4: Download credential files
        self.fileManager.downloadData(self.fileManager.localCredentialSpreadsheet)
        self.fileManager.downloadData(self.fileManager.localCredentialDrive)
        self.credentialSpreadsheet  = self.fileManager.localCredentialSpreadsheet # Rename to make code readable

        # 5: Connect to Google Spreadsheets
        self._authenticateGoogleSpreadSheets() #Creates self.controllerGS
        self._modifyPiGS(error = '')
        
        # 6: Start PiCamera

        self.camera = PiCamera()
        self.camera.resolution = (1296, 972)
        self.camera.framerate = 30
        self.piCamera = 'True'
        
        # 7: Keep track of processes spawned to convert and upload videofiles
        self.processes = [] 

        # 8: Set size of frame
        try:
            self.r
        except AttributeError:
            self.r = (0,0,640,480)

        # 9: Await instructions
        self.monitorCommands()
        
    def __del__(self):
        # Try to close out files and stop running Kinects
        self._modifyPiGS(command = 'None', status = 'Stopped', error = 'UnknownError')
        if self.piCamera:
            if self.camera.recording:
                self.camera.stop_recording()
                self._print('PiCameraStopped: Time=' + str(datetime.datetime.now()) + ', File=Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")

        try:
            if self.device == 'kinect2':
                self.K2device.stop()
            if self.device == 'kinect':
                freenect.sync_stop()
                freenect.shutdown(self.a)
        except AttributeError:
            pass
        self._closeFiles()

    def monitorCommands(self, delta = 10):
        # This function checks the master Controller Google Spreadsheet to determine if a command was issued (delta = seconds to recheck)
        while True:
            self._identifyTank() #Stored in self.tankID
            command, projectID = self._returnCommand()
            if projectID in ['','None']:
                self._reinstructError('ProjectID must be set')
                time.sleep(delta)
                continue

            
            if command != 'None':
                print(command + '\t' + projectID)
                self.fileManager.createProjectData(projectID)    
                self.runCommand(command, projectID)
            self._modifyPiGS(status = 'AwaitingCommand')
            time.sleep(delta)

    def runCommand(self, command, projectID):
        # This function is used to run a specific command found int he  master Controller Google Spreadsheet
        self.projectID = projectID

        # Rename files to make code more readable 
        self.projectDirectory = self.fileManager.localProjectDir
        self.loggerFile = self.fileManager.localLogfile
        self.googleErrorFile = self.fileManager.localProjectDir + 'GoogleErrors.txt'
        self.frameDirectory = self.fileManager.localFrameDir
        self.videoDirectory = self.fileManager.localVideoDir
        self.backupDirectory = self.fileManager.localBackupDir

        if command not in self.commands:
            self._reinstructError(command + ' is not a valid command. Options are ' + str(self.commands))
            
        if command == 'Stop':
            
            if self.piCamera:
                if self.camera.recording:
                    self.camera.stop_recording()
                    self._print('PiCameraStopped: Time: ' + str(datetime.datetime.now()) + ',,File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
                    
                    command = ['python3', 'unit_scripts/process_video.py', self.videoDirectory + str(self.videoCounter).zfill(4) + '_vid.h264']
                    command += [str(self.camera.framerate[0]), self.projectID]
                    self._print(command)
                    self.processes.append(subprocess.Popen(command))

            try:
                if self.device == 'kinect2':
                    self.K2device.stop()
                if self.device == 'kinect':
                    freenect.sync_stop()
                    freenect.shutdown(self.a)
            except Exception as e:
                self._googlePrint(e)
                self._print('ErrorStopping kinect')
                
         

            self._closeFiles()

            self._modifyPiGS(command = 'None', status = 'AwaitingCommand')
            return

        if command == 'UploadData':

            self._modifyPiGS(command = 'None')
            self._uploadFiles()
            return
            
        if command == 'LocalDelete':
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
            self._modifyPiGS(command = 'None', status = 'AwaitingCommand')
            return
        
        self._modifyPiGS(command = 'None', status = 'Running', error = '')


        if command == 'New':
            # Project Directory should not exist. If it does, report error
            if os.path.exists(self.projectDirectory):
                self._reinstructError('New command cannot be run if ouput directory already exists. Use Rewrite or Restart')

        if command == 'Rewrite':
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
            os.makedirs(self.projectDirectory)
            
        if command in ['New','Rewrite']:
            self.masterStart = datetime.datetime.now()
            if command == 'New':
                os.makedirs(self.projectDirectory)
            os.makedirs(self.frameDirectory)
            os.makedirs(self.videoDirectory)
            os.makedirs(self.backupDirectory)
            #self._createDropboxFolders()
            self.frameCounter = 1
            self.videoCounter = 1

        if command == 'Restart':
            logObj = LP(self.loggerFile)
            self.masterStart = logObj.master_start
            #self.r = logObj.bounding_shape
            self.frameCounter = logObj.lastFrameCounter + 1
            self.videoCounter = logObj.lastVideoCounter + 1
            if self.system != logObj.system or self.device != logObj.device or self.piCamera != logObj.camera:
                self._reinstructError('Restart error. System, device, or camera does not match what is in logfile')
                
        self.lf = open(self.loggerFile, 'a', buffering = 1) # line buffered
        self.g_lf = open(self.googleErrorFile, 'a', buffering = 1)
        self._modifyPiGS(start = str(self.masterStart))

        if command in ['New', 'Rewrite']:
            self._print('MasterStart: System: '+self.system + ',,Device: ' + self.device + ',,Camera: ' + str(self.piCamera) + ',,Uname: ' + str(platform.uname()) + ',,TankID: ' + self.tankID + ',,ProjectID: ' + self.projectID)
            self._print('MasterRecordInitialStart: Time: ' + str(self.masterStart))
            self._print('PrepFiles: FirstDepth: PrepFiles/FirstDepth.npy,,LastDepth: PrepFiles/LastDepth.npy,,PiCameraRGB: PiCameraRGB.jpg,,DepthRGB: DepthRGB.jpg')
            picamera_settings = {'AnalogGain': str(self.camera.analog_gain), 'AWB_Gains': str(self.camera.awb_gains), 
                                'AWB_Mode': str(self.camera.awb_mode), 'Brightness': str(self.camera.brightness), 
                                'ClockMode': str(self.camera.clock_mode), 'Contrast': str(self.camera.contrast),
                                'Crop': str(self.camera.crop),'DigitalGain': str(self.camera.digital_gain),
                                'ExposureCompensation': str(self.camera.exposure_compensation),'ExposureMode': str(self.camera.exposure_mode),
                                'ExposureSpeed': str(self.camera.exposure_speed),'FrameRate': str(self.camera.framerate),
                                'ImageDenoise': str(self.camera.image_denoise),'MeterMode': str(self.camera.meter_mode),
                                'RawFormat': str(self.camera.raw_format), 'Resolution': str(self.camera.resolution),
                                'Saturation': str(self.camera.saturation),'SensorMode': str(self.camera.sensor_mode),
                                'Sharpness': str(self.camera.sharpness),'ShutterSpeed': str(self.camera.shutter_speed),
                                'VideoDenoise': str(self.camera.video_denoise),'VideoStabilization': str(self.camera.video_stabilization)}
            self._print('PiCameraSettings: ' + ',,'.join([x + ': ' + picamera_settings[x] for x in sorted(picamera_settings.keys())]))
            #self._createROI(useROI = False)

        else:
            self._print('MasterRecordRestart: Time: ' + str(datetime.datetime.now()))

            
        # Start kinect
        self._start_kinect()
        
        # Diagnose speed
        self._diagnose_speed()

        # Capture data
        self.captureFrames()
    
    def captureFrames(self, frame_delta = 5, background_delta = 5):

        current_background_time = datetime.datetime.now()
        current_frame_time = current_background_time + datetime.timedelta(seconds = 60 * frame_delta)

        command = ''
        
        while True:
            self._modifyPiGS(command = 'None', status = 'Running', error = '')
            # Grab new time
            now = datetime.datetime.now()
            
            # Fix camera if it needs to be
            if self.piCamera:
                if self._video_recording() and not self.camera.recording:
                    self.camera.capture(self.videoDirectory + str(self.videoCounter).zfill(4) + "_pic.jpg")
                    self._print('PiCameraStarted: FrameRate: ' + str(self.camera.framerate) + ',,Resolution: ' + str(self.camera.resolution) + ',,Time: ' + str(datetime.datetime.now()) + ',,VideoFile: Videos/' + str(self.videoCounter).zfill(4) + '_vid.h264,,PicFile: Videos/' + str(self.videoCounter).zfill(4) + '_pic.jpg')
                    self.camera.start_recording(self.videoDirectory + str(self.videoCounter).zfill(4) + "_vid.h264", bitrate=7500000)
                elif not self._video_recording() and self.camera.recording:
                    self._print('PiCameraStopped: Time: ' + str(datetime.datetime.now()) + ',, File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
                    self.camera.stop_recording()
                    #self._print(['rclone', 'copy', self.videoDirectory + str(self.videoCounter).zfill(4) + "_vid.h264"])
                    command = ['python3', 'unit_scripts/process_video.py', self.videoDirectory + str(self.videoCounter).zfill(4) + '_vid.h264']
                    command += [str(self.camera.framerate[0]), self.projectID]
                    self._print(command)
                    self.processes.append(subprocess.Popen(command))
                    self.videoCounter += 1

            # Capture a frame and background if necessary
            
            if now > current_background_time:
                if command == 'Snapshots':
                    out = self._captureFrame(current_frame_time, snapshots = True)
                else:
                    out = self._captureFrame(current_frame_time)
                if out is not None:
                    current_background_time += datetime.timedelta(seconds = 60 * background_delta)
                subprocess.Popen(['python3', 'unit_scripts/drive_updater.py', self.loggerFile])
            else:
                if command == 'Snapshots':
                    out = self._captureFrame(current_frame_time, snapshots = True)
                else:    
                    out = self._captureFrame(current_frame_time, stdev_threshold = stdev_threshold)
            current_frame_time += datetime.timedelta(seconds = 60 * frame_delta)

            self._modifyPiGS(status = 'Running')

            
            # Check google doc to determine if recording has changed.
            try:
                command, projectID = self._returnCommand()
            except KeyError:
                continue                
            if command != 'None':
                if command == 'Snapshots':
                    self._modifyPiGS(command = 'None', status = 'Writing Snapshots')
                    continue
                else:
                    break
            else:
                self._modifyPiGS(error = '')

    def _authenticateGoogleSpreadSheets(self):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentialSpreadsheet, scope)
        for i in range(0,3): # Try to autheticate three times before failing
            try:
                gs = gspread.authorize(credentials)
            except Exception as e:
                self._googlePrint(e)
                continue
            try:
                self.controllerGS = gs.open('Controller')
                pi_ws = self.controllerGS.worksheet('RaspberryPi')
            except Exception as e:
                self._googlePrint(e)
                continue
            try:
                headers = pi_ws.row_values(1)
            except Exception as e:
                self._googlePrint(e)
                continue
            column = headers.index('RaspberryPiID') + 1
            try:
                pi_ws.col_values(column).index(platform.node())
                return True
            except ValueError as e:
                self._googlePrint(e)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                try:
                    pi_ws.append_row([platform.node(),ip,'','','','','','None','Stopped','Error: Awaiting assignment of TankID',str(datetime.datetime.now())])
                except Exception as e:
                    self._googlePrint(e)
                    continue
                return True
            except Exception as e:
                self._googlePrint(e)
                continue    
            time.sleep(2)
        return False
            
    def _identifyDevice(self):
        try:
            global freenect
            import freenect
            self.a = freenect.init()
            if freenect.num_devices(self.a) == 0:
                kinect = False
            elif freenect.num_devices(self.a) > 1:
                self._initError('Multiple Kinect1s attached. Unsure how to handle')
            else:
                kinect = True
        except ImportError:
            kinect = False

        try:
            global rs
            import pyrealsense2 as rs

            ctx = rs.context()
            if len(ctx.devices) == 0:
                realsense = False
            if len(ctx.devices) > 1:
                self._initError('Multiple RealSense devices attached. Unsure how to handle')
            else:
                realsense = True
        except ImportError:
            realsense = False

        if kinect and realsense:
            self._initError('Kinect1 and RealSense devices attached. Unsure how to handle')
        elif not kinect and not realsense:
            self._initError('No depth sensor attached')
        elif kinect:
            self.device = 'kinect'
        else:
            self.device = 'realsense'
       
    def _identifyTank(self):
        while True:
            self._authenticateGoogleSpreadSheets() # link to google drive spreadsheet stored in self.controllerGS 
            pi_ws = self.controllerGS.worksheet('RaspberryPi')
            headers = pi_ws.row_values(1)
            raPiID_col = headers.index('RaspberryPiID') + 1
            for i in range(5):
                try:
                    row = pi_ws.col_values(raPiID_col).index(platform.node()) + 1
                    break
                except Exception as e:
                    self._googlePrint(e)
                    continue
            col = headers.index('TankID')
            if pi_ws.row_values(row)[col] not in ['None','']:
                self.tankID = pi_ws.row_values(row)[col]
                for i in range(5):
                    try:
                        self._modifyPiGS(capability = 'Device=' + self.device + ',Camera=' + str(self.piCamera), status = 'AwaitingCommand')
                        return
                    except Exception as e:
                        self._googlePrint(e)
                        continue
                return
            else:
                self._modifyPiGS(error = 'Awaiting assignment of TankID')
                time.sleep(5)
        
    def _initError(self, message):
        try:
            self._modifyPiGS(command = 'None', status = 'Stopped', error = 'InitError: ' + message)
        except Exception as e:
            self._googlePrint(e)
            pass
        self._print('InitError: ' + message)
        raise TypeError
            
    def _reinstructError(self, message):
        self._modifyPiGS(command = 'None', status = 'AwaitingCommands', error = 'InstructError: ' + message)

        # Update google doc to indicate error
        self.monitorCommands()
 
    def _print(self, text):
        temperature = subprocess.run(['/opt/vc/bin/vcgencmd','measure_temp'], capture_output = True)
        try:
            print(str(text) + ',,Temp: ' + str(temperature.stdout), file = self.lf, flush = True)
        except Exception as e:
            pass
        print(str(text) + ',,Temp: ' + str(temperature.stdout), file = sys.stderr, flush = True)

    def _googlePrint(self, e):
        try:
            print(str(datetime.datetime.now()) + ': ' + str(type(e)) + ': ' + str(e), file = self.g_lf, flush = True)
        except AttributeError as e2: # log file not created yet so just print to stderr
            print(str(datetime.datetime.now()) + ': ' + str(type(e)) + ': ' + str(e), flush = True)

    def _returnRegColor(self, crop = True):
        # This function returns a registered color array
        if self.device == 'kinect':
            out = freenect.sync_get_video()[0]
            
        if self.device == 'realsense':
            frames = self.pipeline.wait_for_frames(1000)
            color_frame = frames.get_color_frame()
            out = np.asanyarray(color_frame.get_data())

        if crop:
            return out[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
        else:
            return out
            
    def _returnDepth(self):
        # This function returns a float64 npy array containing one frame of data with all bad data as NaNs
        if self.device == 'kinect':
            data = freenect.sync_get_depth()[0].astype('float64')
            data[data == 2047] = np.nan # 2047 indicates bad data from Kinect 
            return data[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
        
        if self.device == 'realsense':
            depth_frame = self.pipeline.wait_for_frames(1000).get_depth_frame().as_depth_frame()
            data = np.asanyarray(depth_frame.data)*depth_frame.get_units() # Convert to meters
            data[data==0] = np.nan # 0 indicates bad data from RealSense
            data[data>1] = np.nan # Anything further away than 1 m is a mistake
            return data[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]

    def _returnCommand(self):
        if not self._authenticateGoogleSpreadSheets():
            raise KeyError
            # link to google drive spreadsheet stored in self.controllerGS
        while True:
            try:
                pi_ws = self.controllerGS.worksheet('RaspberryPi')
                headers = pi_ws.row_values(1)
                piIndex = pi_ws.col_values(headers.index('RaspberryPiID') + 1).index(platform.node())
                command = pi_ws.col_values(headers.index('Command') + 1)[piIndex]
                projectID = pi_ws.col_values(headers.index('ProjectID') + 1)[piIndex]
                return command, projectID
            except gspread.exceptions.RequestError:
                continue

    def _modifyPiGS(self, start = None, command = None, status = None, IP = None, capability = None, error = None):
        while not self._authenticateGoogleSpreadSheets(): # link to google drive spreadsheet stored in self.controllerGS
            continue
        try:
            pi_ws = self.controllerGS.worksheet('RaspberryPi')
            headers = pi_ws.row_values(1)
            row = pi_ws.col_values(headers.index('RaspberryPiID')+1).index(platform.node()) + 1
            if start is not None:
                column = headers.index('MasterStart') + 1
                pi_ws.update_cell(row, column, start)
            if command is not None:
                column = headers.index('Command') + 1
                pi_ws.update_cell(row, column, command)
            if status is not None:
                column = headers.index('Status') + 1
                pi_ws.update_cell(row, column, status)
            if error is not None:
                column = headers.index('Error') + 1
                pi_ws.update_cell(row, column, error)
            if IP is not None:
                column = headers.index('IP')+1
                pi_ws.update_cell(row, column, IP)
            if capability is not None:
                column = headers.index('Capability')+1
                pi_ws.update_cell(row, column, capability)
            column = headers.index('Ping') + 1
            pi_ws.update_cell(row, column, str(datetime.datetime.now()))
        except gspread.exceptions.RequestError as e:
            self._print('GoogleError: Time: ' + str(datetime.datetime.now()) + ',,Error: ' + str(e))
        except TypeError:
            self._print('GoogleError: Time: ' + str(datetime.datetime.now()) + ',,Error: Unknown. Gspread does not handle RequestErrors properly...' + str(e))
    
    def _video_recording(self):
        if datetime.datetime.now().hour >= 8 and datetime.datetime.now().hour <= 18:
            return True
        else:
            return False
            
    def _start_kinect(self):
        if self.device == 'kinect':
            freenect.sync_get_depth() #Grabbing a frame initializes the device
            freenect.sync_get_video()

        elif self.device == 'realsense':
            # Create a context object. This object owns the handles to all connected realsense devices
            self.pipeline = rs.pipeline()

            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.depth, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

            # Start streaming
            self.pipeline.start(config)
        
            frames = self.pipeline.wait_for_frames(1000)
            depth = frames.get_depth_frame()
            self.r = (0,0,depth.width,depth.height)

    def _diagnose_speed(self, time = 10):
        print('Diagnosing speed for ' + str(time) + ' seconds.', file = sys.stderr)
        delta = datetime.timedelta(seconds = time)
        start_t = datetime.datetime.now()
        counter = 0
        while True:
            depth = self._returnDepth()
            counter += 1
            if datetime.datetime.now() - start_t > delta:
                break
        #Grab single snapshot of depth and save it
        depth = self._returnDepth()
        np.save(self.projectDirectory +'Frames/FirstFrame.npy', depth)

        #Grab a bunch of depth files to characterize the variability
        data = np.zeros(shape = (50, self.r[3], self.r[2]))
        for i in range(0, 50):
            data[i] = self._returnDepth()
            
        counts = np.count_nonzero(~np.isnan(data), axis = 0)
        std = np.nanstd(data, axis = 0)
        np.save(self.projectDirectory +'Frames/FirstDataCount.npy', counts)
        np.save(self.projectDirectory +'Frames/StdevCount.npy', std)
         
        self._print('DiagnoseSpeed: Rate: ' + str(counter/time))

        self._print('FirstFrameCaptured: FirstFrame: Frames/FirstFrame.npy,,GoodDataCount: Frames/FirstDataCount.npy,,StdevCount: Frames/StdevCount.npy')
    
    def _captureFrame(self, endtime, max_frames = 40, stdev_threshold = 20, snapshots = False):
        # Captures time averaged frame of depth data
        sums = np.zeros(shape = (self.r[3], self.r[2]))
        n = np.zeros(shape = (self.r[3], self.r[2]))
        stds = np.zeros(shape = (self.r[3], self.r[2]))
        
        current_time = datetime.datetime.now()
        if current_time >= endtime:
            self._print('Frame without data')
            return

        counter = 1

        all_data = np.empty(shape = (int(max_frames), self.r[3], self.r[2]))
        all_data[:] = np.nan
        
        for i in range(0, max_frames):
            all_data[i] = self._returnDepth()
            current_time = datetime.datetime.now()

            if snapshots:
                self._print('SnapshotCaptured: NpyFile: Frames/Snapshot_' + str(counter).zfill(6) + '.npy,,Time: ' + str(current_time)  + ',,GP: ' + str(np.count_nonzero(~np.isnan(all_data[i]))))
                np.save(self.projectDirectory +'Frames/Snapshot_' + str(counter).zfill(6) + '.npy', all_data[i])

            
            counter += 1

            if current_time >= endtime:
                break
            time.sleep(10)
        
        med = np.nanmean(all_data, axis = 0)
        
        std = np.nanstd(all_data, axis = 0)
        
        med[np.isnan(std)] = np.nan

        med[std > stdev_threshold] = np.nan
        std[std > stdev_threshold] = np.nan

        counts = np.count_nonzero(~np.isnan(all_data), axis = 0)

        med[counts < 3] = np.nan
        std[counts < 3] = np.nan

        color = self._returnRegColor()                        
        
        self._print('FrameCaptured: NpyFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy,,PicFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg,,Time: ' + str(endtime)  + ',,NFrames: ' + str(i) + ',,AvgMed: '+ '%.2f' % np.nanmean(med) + ',,AvgStd: ' + '%.2f' % np.nanmean(std) + ',,GP: ' + str(np.count_nonzero(~np.isnan(med))))
        
        np.save(self.projectDirectory +'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy', med)
        matplotlib.image.imsave(self.projectDirectory+'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg', color)
        
        self.frameCounter += 1

        return med

            
    def _uploadFiles(self):
        self._modifyPiGS(status = 'Finishing converting and uploading of videos')
        for p in self.processes:
            p.communicate()
        
        for movieFile in os.listdir(self.videoDirectory):
            if '.h264' in movieFile:
                command = ['python3', 'unit_scripts/process_video.py', self.videoDirectory + movieFile]
                command += [str(self.camera.framerate[0]), self.projectID]
                self._print(command)
                self.processes.append(subprocess.Popen(command))

        for p in self.processes:
            p.communicate()

        self._modifyPiGS(status = 'Creating prep files')

        # Move files around as appropriate
        prepDirectory = self.projectDirectory + 'PrepFiles/'
        shutil.rmtree(prepDirectory) if os.path.exists(prepDirectory) else None
        os.makedirs(prepDirectory)

        lp = LP(self.loggerFile)

        self.frameCounter = lp.lastFrameCounter + 1

        videoObj = [x for x in lp.movies if x.startTime.hour >= 8 and x.startTime.hour <= 20][0]
        subprocess.call(['cp', self.projectDirectory + videoObj.pic_file, prepDirectory + 'PiCameraRGB.jpg'])

        subprocess.call(['cp', self.projectDirectory + lp.movies[-1].pic_file, prepDirectory + 'LastPiCameraRGB.jpg'])

        # Find depthfile that is closest to the video file time
        depthObj = [x for x in lp.frames if x.time > videoObj.startTime][0]


        subprocess.call(['cp', self.projectDirectory + depthObj.pic_file, prepDirectory + 'DepthRGB.jpg'])

        if not os.path.isdir(self.frameDirectory):
            self._modifyPiGS(status = 'Error: ' + self.frameDirectory + ' does not exist.')
            return

        subprocess.call(['cp', self.frameDirectory + 'Frame_000001.npy', prepDirectory + 'FirstDepth.npy'])
        subprocess.call(['cp', self.frameDirectory + 'Frame_' + str(self.frameCounter-1).zfill(6) + '.npy', prepDirectory + 'LastDepth.npy'])
        
        try:
            self._modifyPiGS(status = 'Uploading data to cloud')
            self.fileManager.uploadData(self.frameDirectory, tarred = True)
            #print(prepDirectory)
            self.fileManager.uploadData(prepDirectory)
            #print(self.videoDirectory)
            self.fileManager.uploadData(self.videoDirectory)
            #print(self.loggerFile)
            self.fileManager.uploadData(self.loggerFile)
            self._modifyPiGS(error = 'UploadSuccessful, ready for delete')

        except Exception as e:
            print('UploadError: ' + str(e))
            self._modifyPiGS(error = 'UploadFailed, Need to rerun')
            raise Exception
        
    def _closeFiles(self):
       try:
            self._print('MasterRecordStop: ' + str(datetime.datetime.now()))
            self.lf.close()
       except AttributeError:
           pass
       try:
           if self.system == 'mac':
               self.caff.kill()
       except AttributeError:
           pass

