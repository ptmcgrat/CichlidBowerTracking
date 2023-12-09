import subprocess, os, pdb


class ClusterPreparer():
    # This class takes in directory information and a logfile containing depth information and performs the following:
    # 1. Identifies tray using manual input
    # 2. Interpolates and smooths depth data
    # 3. Automatically identifies bower location
    # 4. Analyze building, shape, and other pertinent info of the bower

    def __init__(self, fileManager, workers):

        self.__version__ = '1.0.0'

        self.fileManager = fileManager
        self.workers = workers

    def downloadProjectData(self):
        self.fileManager.createDirectory(self.fileManager.localLogfileDir)
        self.fileManager.createDirectory(self.fileManager.localMasterDir)
        self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
        self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
        self.fileManager.createDirectory(self.fileManager.localTempDir)
        self.fileManager.createDirectory(self.fileManager.localAllClipsDir)
        self.fileManager.createDirectory(self.fileManager.localManualLabelClipsDir)
        self.fileManager.createDirectory(self.fileManager.localManualLabelFramesDir)
        #self.createDirectory(self.localPaceDir)

        self.fileManager.downloadData(self.fileManager.localLogfile)
        print('Downloading video ' + self.fileManager.localVideoDir)
        self.fileManager.downloadData(self.fileManager.localVideoDir)


    def validateInputData(self):
        
        assert os.path.exists(self.fileManager.localTroubleshootingDir)
        assert os.path.exists(self.fileManager.localAnalysisDir)
        assert os.path.exists(self.fileManager.localTempDir)
        assert os.path.exists(self.fileManager.localAllClipsDir)
        assert os.path.exists(self.fileManager.localManualLabelClipsDir)
        assert os.path.exists(self.fileManager.localManualLabelFramesDir)
        assert os.path.exists(self.fileManager.localLogfileDir)

    def runClusterAnalysis(self, videoIndex):
        self.videoObj = self.fileManager.returnVideoObject(videoIndex)
        assert os.path.exists(self.videoObj.localVideoFile)


        command = ['python3', 'VideoFocus.py']
        command.extend(['--Movie_file', self.videoObj.localVideoFile])
        command.extend(['--Video_framerate', str(self.videoObj.framerate)])
        command.extend(['--Num_workers', str(self.workers)])
        command.extend(['--Log', self.videoObj.localLogfile])
        command.extend(['--HMM_temp_directory', self.videoObj.localTempDir])
        command.extend(['--HMM_filename', self.videoObj.localHMMFile])
        command.extend(['--HMM_transition_filename', self.videoObj.localRawCoordsFile])
        command.extend(['--Cl_labeled_transition_filename', self.videoObj.localLabeledCoordsFile])
        command.extend(['--Cl_labeled_cluster_filename', self.videoObj.localLabeledClustersFile])
        command.extend(['--Cl_videos_directory', self.videoObj.localAllClipsDir])
        command.extend(['--ML_frames_directory', self.videoObj.localManualLabelFramesDir])
        command.extend(['--ML_videos_directory', self.videoObj.localManualLabelClipsDir])
        command.extend(['--Video_start_time', str(self.videoObj.startTime)])
        command.extend(['--VideoID', self.videoObj.baseName])

        if not os.path.isdir('CichlidActionDetection'):
            subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/CichlidActionDetection'])

        os.chdir('CichlidActionDetection')
        subprocess.run(['git', 'pull'])
        subprocess.run(command)
        os.chdir('..')



