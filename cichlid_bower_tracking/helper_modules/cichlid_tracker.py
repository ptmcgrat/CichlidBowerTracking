import platform, sys, os, shutil, datetime, subprocess, pdb, time, sendgrid, psutil, csv
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
from cichlid_bower_tracking.helper_modules.log_parser import LogParser as LP
from cichlid_bower_tracking.helper_modules.googleController import GoogleController as GC
import pandas as pd
from picamera import PiCamera
import numpy as np
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import *

import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import matplotlib.image

sys.path.append(sys.path[0] + '/unit_scripts')
sys.path.append(sys.path[0] + '/helper_modules')

class CichlidTracker:
    def __init__(self):

        # 1: Define valid commands and ignore warnings
        self.commands = ['New', 'Restart', 'Stop', 'Rewrite', 'UploadData', 'LocalDelete','DeleteEntireProject']
        np.seterr(invalid='ignore')

        # 2: Determine which depth sensor is attached (This script can handle DepthSense cameras)
        self._identifyDevice() #Stored in self.device
        self.system = platform.node()

        # 3: Create file manager
        self.fileManager = FM()

        # 4: Start PiCamera
        self.camera = PiCamera()
        self.camera.resolution = (1296, 972)
        self.camera.framerate = 30
        self.piCamera = 'True'

        # 5: Download credential files
        self.fileManager.downloadData(self.fileManager.localCredentialDir)
        self.googleController = GC(self.fileManager.localCredentialSpreadsheet)
        self.tankID = self.googleController.tankID
        self.googleController.modifyPiGS('Capability', 'Device=' + self.device + ',Camera=' + str(self.piCamera), ping = True)
        self.googleController.modifyPiGS('Error', '', ping = False)

        # 6: Keep track of processes spawned to convert and upload videofiles
        self.processes = [] 

        # 7: Set size of frame
        try:
            self.r
        except AttributeError:
            self.r = (0,0,640,480)

        # 8: Create email server
        with open(self.fileManager.localEmailCredentialFile) as f:
            my_api_key = [x.strip() for x in f.readlines()][0]

        self.sg = sendgrid.SendGridAPIClient(api_key=my_api_key)
        #self.personalization = sendgrid.Personalization()
        #self.personalization.add_to(sendgrid.To('pmcgrath7@gatech.edu'))
        #for email in ['bshi42@gatech.edu', 'bhegarty6@gatech.edu', 'zjohnson37@gatech.edu']:
            #self.personalization.add_bcc(sendgrid.Bcc(email))
            
        # 9: Declare class variable for rs running flag
        self.is_rs_running = False
        # 10: Await instructions
        print('Monitoring commands')
        self.running = False

        self.monitorCommands()
        
    def __del__(self):
        # Try to close out files and stop running Kinects
        self.googleController.modifyPiGS('Command','None', ping = False)
        self.googleController.modifyPiGS('Status','Stopped', ping = False)
        self.googleController.modifyPiGS('Error','UnknownError', ping = False)

        if self.running:
            #new_email = sendgrid.Mail(
                #from_email='themcgrathlab@gmail.com', 
                #subject= self.tankID + ' has stopped running', 
                #html_content= 'Check the Controller sheet'
            #)
            #new_email.add_personalization(self.personalization)

            # Get a JSON-ready representation of the Mail object
            # Send an HTTP POST request to /mail/send
            #response = self.sg.send(new_email)

            current_temp = psutil.sensors_temperatures()['cpu_thermal'][0][1]
            harddrive_use = psutil.disk_usage(self.fileManager.localMasterDir)[3]
            cpu_use = psutil.cpu_percent()
            ram_use = psutil.virtual_memory()[2]

            self._print('UnknownExceptionExit: Temperature: ' + str(current_temp) + ',,HardDriveUsage: ' + str(harddrive_use) + ',,CPUUsage: ' + str(cpu_use) + ',,RAMUse: ' + str(ram_use))



        if self.piCamera:
            if self.camera.recording:
                self.camera.stop_recording()
                self._print('PiCameraStopped: Time=' + str(datetime.datetime.now()) + ',,File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")


        if self.device == 'realsense':
            try:
                self.pipeline.stop()
            except:
                pass

        self._closeFiles()

    def monitorCommands(self, delta = 20):
        # This function checks the master Controller Google Spreadsheet to determine if a command was issued (delta = seconds to recheck)
        self.googleController.modifyPiGS('Status', 'AwaitingCommand')
        #self.googleController.modifyPiGS('Error', '', ping = False)

        while True:
            command, projectID, analysisID = self._returnCommand()
            if projectID in ['','None']:
                self._reinstructError('ProjectID must be set')
                time.sleep(delta)
                continue
            
            elif analysisID in ['','None']:
                self._reinstructError('AnalysisID must be set')
                time.sleep(delta)
                continue

            if command not in ['None',None]:
                self.googleController.modifyPiGS('Error', '', ping = False)

                print(command + '\t' + projectID + '\t' + analysisID)
                self.fileManager = FM(analysisID = analysisID, projectID = projectID)
                self.projectID = projectID
                
                # Make sure Analysis File exists and is good

                if not self.fileManager.checkFileExists(self.fileManager.localSummaryFile):
                    self._reinstructError('Must create Analysis States csv file before running: ' + self.fileManager.localSummaryFile.split('Elements')[1])
                self.fileManager.downloadData(self.fileManager.localSummaryFile)
                s_dt = pd.read_csv(self.fileManager.localSummaryFile)
                if 'projectID' not in s_dt.columns:
                    self._reinstructError('projectID column most be in analysis states csv file')

                self.analysisID = analysisID
                self.runCommand(command)

            time.sleep(delta)

    def runCommand(self, command):
        # This function is used to run a specific command found in the  master Controller Google Spreadsheet

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
            
            self.running = False
            if self.piCamera:
                if self.camera.recording:
                    self.camera.stop_recording()
                    self._print('PiCameraStopped: Time: ' + str(datetime.datetime.now()) + ',,File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
                    
                    #command = ['python3', 'unit_scripts/process_video.py', self.videoDirectory + str(self.videoCounter).zfill(4) + '_vid.h264']
                    #command += [str(self.camera.framerate[0]), self.projectID, self.analysisID]
                    #self._print(command)
                    #self.processes.append(subprocess.Popen(command))

            try:
                if self.device == 'realsense':
                    self.pipeline.stop()
            except Exception as e:
                self._print('ErrorStopping kinect: ' + str(e))
                
            self._closeFiles()

            self.googleController.modifyPiGS('Command', 'None', ping = False)
            self.googleController.modifyPiGS('Status', 'AwaitingCommand', ping = False)
            return

        if command == 'UploadData':

            self.googleController.modifyPiGS('Command', 'None')
            self._uploadFiles()
            return
            
        if command == 'LocalDelete':
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
            self.googleController.modifyPiGS('Command', 'None', ping = False)
            self.googleController.modifyPiGS('Status', 'AwaitingCommand', ping = False)
            return

        if command == 'DeleteEntireProject':
            self.googleController.modifyPiGS('Status', 'DeletingData', ping = False)
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
            self.fileManager.deleteCloudData(self.projectDirectory)
            self.googleController.modifyPiGS('Command', 'None', ping = False)
            self.googleController.modifyPiGS('Status', 'AwaitingCommand', ping = False)

            return            

        self.googleController.modifyPiGS('Command', 'None', ping = False)
        self.googleController.modifyPiGS('Status', 'Running', ping = False)
        self.googleController.modifyPiGS('Error', '', ping = False)


        if command == 'New':
            # Project Directory should not exist. If it does, report error
            if os.path.exists(self.projectDirectory):
                self._reinstructError('New command cannot be run if ouput directory already exists on the pi. Use Rewrite or Restart')
            if self.fileManager.checkFileExists(self.projectDirectory):
                self._reinstructError('New command cannot be run if ouput directory already exists on the cloud. Did you forget to rename the project? If not delete the data on Dropbox')

        if command == 'Rewrite':
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
            if self.fileManager.checkFileExists(self.projectDirectory):
                self._reinstructError('Rewrite command cannot be run if ouput directory already exists on the cloud. Delete the data on Dropbox')
            
        if command in ['New','Rewrite']:
            self.masterStart = datetime.datetime.now()
            
            os.makedirs(self.projectDirectory)
            os.makedirs(self.frameDirectory)
            os.makedirs(self.videoDirectory)
            os.makedirs(self.backupDirectory)

            self.googleController.addProjectID(self.projectID, self.googleErrorFile)
            
            #self._createDropboxFolders()
            self.frameCounter = 1
            self.videoCounter = 1
            self.firstDepthCaptured = False

        if command == 'Restart':
            logObj = LP(self.loggerFile)
            self.masterStart = logObj.master_start
            self.frameCounter = logObj.lastFrameCounter + 1
            self.videoCounter = logObj.lastVideoCounter + 1
            if self.system != logObj.system or self.device != logObj.device:
                self._reinstructError('Restart error. LogData: ' + ','.join([str(x) for x in [logObj.system,logObj.device,logObj.camera]]) + ',, SystemData: ' + ','.join([str(x) for x in [self.system, self.device, self.camera]]))
                return
            if self.device != 'None':
                subprocess.Popen(['python3', 'unit_scripts/drive_updater.py', self.loggerFile])
                pass

        self.lf = open(self.loggerFile, 'a', buffering = 1) # line buffered
        self.googleController.modifyPiGS('MasterStart',str(self.masterStart), ping = False)

        if command in ['New', 'Rewrite']:
            self._print('MasterStart: System: '+self.system + ',,Device: ' + self.device + ',,Camera: ' + str(self.piCamera) + ',,Uname: ' + str(platform.uname()) + ',,TankID: ' + self.tankID + ',,ProjectID: ' + self.projectID + ',,AnalysisID: ' + self.analysisID)
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
        if self.device != 'None':
            self._start_kinect()
            # Diagnose speed
            self._diagnose_speed()

        # Capture data
        self.running = True
        self.captureFrames()
    
    def captureFrames(self, frame_delta = 5, background_delta = 5):

        current_background_time = datetime.datetime.now()
        current_frame_time = current_background_time + datetime.timedelta(seconds = 60 * frame_delta)

        command = ''
        
        while True:

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
                    command += [str(self.camera.framerate[0]), self.projectID, self.analysisID]
                    self._print(command)
                    self.processes.append(subprocess.Popen(command))
                    self.videoCounter += 1

            # Capture a frame and background if necessary
            day_of_week = now.isocalendar().weekday
            #the rs needs to alternatively turn on at 6 am and 7:45
            # 1 - Monday, 2 - Tuesday, 3 - Wednesday, 4- Thursday, 5 - Friday, 6 - Saturday, 7 - Sunday
            if (day_of_week % 2 == 0 and now.hour < 6) or (day_of_week % 2 != 0 and ((now.hour < 7) or (now.hour == 7 and now.minute <= 45))) or (now.hour >= 18) or (now.hour == 17 and now.minute >= 50):
                if self.is_rs_running:
                    try:
                        self.pipeline.stop()
                    except:
                        print('pipeline not running')
                    self.is_rs_running = False
                self.googleController.modifyPiGS('Error','Realsense Off', ping= True)
                time.sleep(300)    
            else:
                if not self.is_rs_running:
                    try:
                        self._start_kinect()
                    except:
                        print('pipeline already running')
                    self.is_rs_running = True
                if self.device != 'None':
                    if now > current_background_time:
                        out = self._captureFrame(current_frame_time)
                        if out is not None:
                           current_background_time += datetime.timedelta(seconds = 60 * background_delta)
                        subprocess.Popen(['python3', 'unit_scripts/drive_updater.py', self.loggerFile])
                    else:
                        logObj = LP(self.fileManager.localLogfile)
                        out = self._captureFrame(current_frame_time, stdev_threshold = stdev_threshold)
                        if len(logObj.frames) > 1 and logObj.frames[-1].std< 0.00001 and logObj.frames[-1].gp==logObj.frames[-2].gp:
                            self.reboot_rs('Duplicate Datapoint')
                            self.captureFrames()
                else:
                    while datetime.datetime.now() < current_frame_time:
                        time.sleep(5)

            current_frame_time += datetime.timedelta(seconds = 60 * frame_delta)

            # Check google doc to determine if recording has changed.
            try:
                command, projectID, analysisID = self._returnCommand()
            except KeyError:
                continue
            if command == 'TankResetStart':
                self._print('TankResetStart: Time: ' + str(datetime.datetime.now()))
                self.googleController.modifyPiGS('Command', 'None', ping = False)

            elif command == 'TankResetStop':
                self._print('TankResetStop: Time: ' + str(datetime.datetime.now()))
                self.googleController.modifyPiGS('Command', 'None', ping = False)

            elif command != 'None' and command is not None:
                break
            else:
                self.googleController.modifyPiGS('Error', '')

    def _identifyDevice(self):

        try:
            global rs
            import pyrealsense2 as rs

            ctx = rs.context()
            if len(ctx.devices) == 0:
                self.device = 'None'
            elif len(ctx.devices) > 1:
                self._initError('Multiple RealSense devices attached. Unsure how to handle')
            else:
                self.device = 'realsense'
        except Exception:
            self.device = 'None'


    def _initError(self, message):
        self.googleController.modifyPiGS('Command', 'None')
        self.googleController.modifyPiGS('Status', 'Stopped', ping = False)
        self.googleController.modifyPiGS('Error', 'InitError: ' + message, ping = False)

        self._print('InitError: ' + message)
        raise TypeError
            
    def _reinstructError(self, message):
        self.googleController.modifyPiGS('Error', 'InstructError: ' + message, ping = False)
        self._print(message)

        # Update google doc to indicate error
        self.monitorCommands()
 
    def _print(self, text):
        #temperature = subprocess.run(['/opt/vc/bin/vcgencmd','measure_temp'], capture_output = True)
        try:
            print(str(text), file = self.lf, flush = True)
        except Exception as e:
            self.lf = open(self.loggerFile, 'a', buffering = 1) # line buffered
            print(str(text), file = self.lf, flush = True)
        print(str(text), file = sys.stderr, flush = True)

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
        if self.device == 'realsense':
            b, frames = self.pipeline.try_wait_for_frames(3000)
            
            if not b:
                self._print('realsense error attempting reboot')
                message = 'ReturnDepth Error'
                self.reboot_rs(message)
                self._returnDepth()
                
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame().as_depth_frame()
            #except RuntimeError:
            #    self._googlePrint('No frame received from Kinect. Restarting')
            #    self._start_kinect()
            #    depth_frame = self.pipeline.wait_for_frames(1000).get_depth_frame().as_depth_frame()

            data = np.asanyarray(depth_frame.data)*depth_frame.get_units()*100 # Convert to centimeters
            data[data==0] = np.nan # 0 indicates bad data from RealSense
            return data[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
    
    def reboot_rs(self,message):  
        if self.device == 'realsense':
            self.write_restart_to_logs(message,'0')
            self.send_email(message + ' The rs is attempting reboot.')
            try:
                self._print('stopping realsense')
                self.pipeline.stop()
            except Exception as e:
                self._print('Error stopping realsense: ' + str(e))
                raise Exception
            try:
                self._print('Starting realsense')
                self._start_kinect()
            except Exception as e:
                self._print('Error starting realsense: ' + str(e))
                raise Exception
            
    # We are currently not rebooting the pi for errors. But this is the code that would do it.
    #def reboot_pi(self,message):
    #    self.write_restart_to_logs(message,'1')
    #    self.send_email(message + ' The pi is attempting reboot.')
    #    self._print('Rebooting Pi : ')
    #    try:
    #        os.system('sudo reboot -h now')
    #    except Exception as e:
    #            self._print('Error rebooting down pi: ' + str(e))
    #            raise Exception

    def send_email(self, reason):
        from_email=Email("themcgrathlab@gmail.com")
        to_email=[To("bshi42@gatech.edu"),To("jtata6@gatech.edu")]
        subject= self.tankID + " encountered an error"
        content=Content("text/plain","The error was "+reason)
        new_email = Mail(from_email,to_email,subject,content)
        response = self.sg.send(new_email)
        self._print(response.status_code)
        self._print(response.body)
        self._print(response.headers)
        
    def write_restart_to_logs(self,reason,isPiReboot):
        restart_time = datetime.datetime.now()
        data = [restart_time, reason, isPiReboot]
        with open(self.fileManager.localPiErrorFile, 'a', encoding='UTF8') as f:
             writer = csv.writer(f)
             writer.writerow(data)


    def _returnCommand(self):

        command, projectID, analysisID = self.googleController.getPiGS(['Command','ProjectID','AnalysisID'])
        return command, projectID, analysisID

    def _video_recording(self, time = None):
        if time is None:
            time = datetime.datetime.now()
        if time.hour >= 8 and time.hour < 18:
            return True
        else:
            return False
            
    def _start_kinect(self):

        if self.device == 'realsense':
            # Create a context object. This object owns the handles to all connected realsense devices
            self.pipeline = rs.pipeline()
            self.align = rs.align(rs.stream.color)
            
            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

            # Start streaming
            self.profile = self.pipeline.start(config)
            #device = self.profile.get_device()
            #depth_sensor = device.first_depth_sensor()
            #device.hardware_reset()
            self.is_rs_running = True
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

        self._print('FirstFrameCaptured: FirstFrame: Frames/FirstFrame.npy,,GoodDataCount: Frames/FirstDataCount.npy,,StdevCount: Frames/StdevCount.npy,,Units: cm')
    
    def _captureFrame(self, endtime, max_frames = 40):
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
            
            counter += 1

            if current_time >= endtime:
                break
            time.sleep(10) # Wait 10 seconds
        
        # Filter out top 3 data points
        mins = np.nanmin(all_data, axis = 0)
        mins[np.nansum((all_data == mins).astype(int), axis = 0) > 3] = np.nan
        all_data[all_data == mins] = np.nan
        mins = np.nanmin(all_data, axis = 0)
        mins[np.nansum((all_data == mins).astype(int), axis = 0) > 3] = np.nan
        all_data[all_data == mins] = np.nan
        mins = np.nanmin(all_data, axis = 0)
        mins[np.nansum((all_data == mins).astype(int), axis = 0) > 3] = np.nan
        all_data[all_data == mins] = np.nan
        std = np.nanstd(all_data, axis = 0)
        med = np.nanmedian(all_data, axis = 0)

        bad_all_pixels = np.count_nonzero(np.isnan(all_data))
        good_all_pixels = np.count_nonzero(~np.isnan(all_data))


        counts = np.count_nonzero(~np.isnan(all_data), axis = 0)

        color = self._returnRegColor()                        
        
        outstring = 'FrameCaptured: NpyFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy,,PicFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg,,'
        outstring += 'Time: ' + str(endtime)  + ',,NFrames: ' + str(i+1) + ',,AvgMed: '+ '%.2f' % np.nanmean(med) + ',,AvgStd: ' + '%.2f' % np.nanmean(std) + ',,'
        outstring += 'GP: ' + str(np.count_nonzero(~np.isnan(med))) + ',,AllPixelsBad: ' + str(bad_all_pixels) + ',,AllPixelsGood: ' + str(good_all_pixels) + ',,'
        outstring += 'LOF: ' + str(self._video_recording(time = endtime)) + ',,'
        outstring += 'Temperature: ' + str(psutil.sensors_temperatures()['cpu_thermal'][0][1]) + ',,'
        outstring += 'HardDriveUsage: ' + str(psutil.disk_usage(self.fileManager.localMasterDir)[3]) + ',,'
        outstring += 'CPUUsage: ' + str(psutil.cpu_percent()) + ',,'
        outstring += 'RAMUsage: ' + str(psutil.virtual_memory()[2])

        self._print(outstring)
        np.save(self.projectDirectory + 'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy', med)
        np.save(self.projectDirectory + 'Frames/Frame_std_' + str(self.frameCounter).zfill(6) + '.npy', std)
        matplotlib.image.imsave(self.projectDirectory+'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg', color)
        
        if self.frameCounter % 1000000 == 0:
            self._print('AllDataCaptured: NpyFile: Frames/AllData_' + str(self.frameCounter).zfill(6) + '.npy,,PicFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg,,Time: ' + str(endtime)  + ',,NFrames: ' + str(i+1))
            np.save(self.projectDirectory +'Frames/AllData_' + str(self.frameCounter).zfill(6) + '.npy', all_data)


        self.frameCounter += 1

        return med

            
    def _uploadFiles(self):
        self.googleController.modifyPiGS('Status', 'Finishing converting and uploading of videos')
        for p in self.processes:
            p.communicate()
        
        for movieFile in os.listdir(self.videoDirectory):
            if '.h264' in movieFile:
                command = ['python3', 'unit_scripts/process_video.py', self.videoDirectory + movieFile]
                command += [str(self.camera.framerate[0]), self.projectID, self.analysisID]
                self._print(command)
                self.processes.append(subprocess.Popen(command))

        for p in self.processes:
            p.communicate()

        self.googleController.modifyPiGS('Status','Creating prep files')

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
        if self.device != 'None':
            # Filter depth objects for those that are in the daytime
            depthObjs = [x for x in lp.frames if (x.time > videoObj.startTime and self._video_recording(time = x.time))]

            subprocess.call(['cp', self.projectDirectory + depthObjs[0].pic_file, prepDirectory + 'FirstDepthRGB.jpg'])
            subprocess.call(['cp', self.projectDirectory + depthObjs[0].npy_file, prepDirectory + 'FirstDepth.npy'])
            subprocess.call(['cp', self.projectDirectory + depthObjs[-1].pic_file, prepDirectory + 'LastDepthRGB.jpg'])
            subprocess.call(['cp', self.projectDirectory + depthObjs[-1].npy_file, prepDirectory + 'LastDepth.npy'])

            if not os.path.isdir(self.frameDirectory):
                self.googleController.modifyPiGS('Status', 'Error: ' + self.frameDirectory + ' does not exist.')
                return
        
        try:
            self.googleController.modifyPiGS('Status', 'Uploading data to cloud')
            if self.device != 'None':
                self.fileManager.uploadData(self.frameDirectory, tarred = True)
            #print(prepDirectory)
            self.fileManager.uploadData(prepDirectory)
            #print(self.videoDirectory)
            self.fileManager.uploadData(self.videoDirectory)
            #print(self.loggerFile)
            self.fileManager.uploadData(self.loggerFile)
            self.fileManager.uploadData(self.googleErrorFile)
            self.googleController.modifyPiGS('Error','UploadSuccessful, ready for delete')
            
            self.fileManager.downloadData(self.fileManager.localSummaryFile)
            s_dt = pd.read_csv(self.fileManager.localSummaryFile)

            if self.projectID not in s_dt.projectID.values:
                s_dt.loc[len(s_dt.index)] = ['']*len(s_dt.columns)
                s_dt.loc[len(s_dt.index) - 1,'projectID'] = self.projectID
                s_dt.to_csv(self.fileManager.localSummaryFile, index = False)
                self.fileManager.uploadData(self.fileManager.localSummaryFile)

            self.googleController.modifyPiGS('Error','UploadSuccessful, ready for delete')


        except Exception as e:
            print('UploadError: ' + str(e))
            self.googleController.modifyPiGS('Error','UploadFailed, Need to rerun')
            raise Exception
        
    def _closeFiles(self):
       try:
            self._print('MasterRecordStop: ' + str(datetime.datetime.now()))
            self.lf.close()
       except AttributeError:
           pass

