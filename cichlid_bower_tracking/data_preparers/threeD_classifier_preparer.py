import subprocess, pickle, os, shutil, pdb, scipy, datetime
from skimage import io
import pandas as pd


class ThreeDClassifierPreparer:
    # This class takes in directory information and a logfile containing depth information and performs the following:
    # 1. Identifies tray using manual input
    # 2. Interpolates and smooths depth data
    # 3. Automatically identifies bower location
    # 4. Analyze building, shape, and other pertinent info of the bower

    def __init__(self, fileManager):
        self.__version__ = '1.0.0'

        self.fileManager = fileManager

    def downloadProjectData(self):
        self.createDirectory(self.localMasterDir)
        self.createDirectory(self.localLogfileDir)

        self.downloadData(self.localLogfile)
        self.downloadData(self.localAllClipsDir, tarred_subdirs = True)
        self.downloadData(self.localAnalysisDir)
        self.downloadData(self.localTroubleshootingDir)
        if self.modelID is not None:
            self.downloadData(self.local3DModelDir)
            #self.createDirectory(self.localPaceDir)

    def validateInputData(self):
        assert os.path.exists(self.fileManager.localAllClipsDir) # Clips
        assert os.path.exists(self.fileManager.localVideoModelFile) # Model
        assert os.path.exists(self.fileManager.localVideoClassesFile) # Classes
        assert os.path.exists(self.fileManager.localModelCommandsFile) # Commands

        videos = list(range(len(self.fileManager.lp.movies)))
        
        for videoIndex in videos:
            videoObj = self.fileManager.returnVideoObject(videoIndex)
            assert os.path.exists(videoObj.localLabeledClustersFile)


    def predictLabels(self):


        # Create mapping from videos to projectID

        with open(self.fileManager.localVideoProjectsFile, 'w') as f:
            print('VideoFile,Label,ProjectID', file = f)

            for videofile in [x for x in os.listdir(self.fileManager.localAllClipsDir) if '.mp4' in x]:
                print(videofile + ',,' + self.fileManager.projectID, file = f)

        # Run command
        command = ['python3', 'PredictLabels.py']
        command.extend(['--Videos_directory', self.fileManager.localAllClipsDir])
        command.extend(['--Videos_file', self.fileManager.localVideoProjectsFile])
        command.extend(['--Trained_model', self.fileManager.localVideoModelFile])
        command.extend(['--Training_log', self.fileManager.localModelCommandsFile])
        command.extend(['--Classes_file', self.fileManager.localVideoClassesFile])
        command.extend(['--Results_directory', self.fileManager.localTempClassifierDir])
        print(' '.join(command))


        if not os.path.isdir('VideoClassifier'):
            subprocess.run(['git', 'clone', 'https://www.github.com/ptmcgrat/VideoClassifier'])

        #command = "source activate CichlidActionClassification; " + ' ' .join(command)
        command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidActionClassification; " + ' '.join(command)
        os.chdir('VideoClassifier')
        subprocess.run(['git', 'pull'])
        subprocess.run('bash -c \"' + command + '\"', shell = True)
        os.chdir('..')
        shutil.copy(os.path.join(self.fileManager.localTempClassifierDir,'TrainingLog.txt'), self.fileManager.localClusterClassificationLogfile)

    def createSummaryFile(self):
        
        for videoIndex, video in enumerate(self.fileManager.lp.movies):
            videoObj = self.fileManager.returnVideoObject(videoIndex)
            new_dt = pd.read_csv(videoObj.localLabeledClustersFile)
            try:
                c_dt = c_dt.append(new_dt)
            except NameError:
                c_dt = new_dt
        pred_dt = pd.read_csv(os.path.join(self.fileManager.localTempClassifierDir,'predictions_0.csv'), index_col = 0)
        pred_dt['ClipName'] = pred_dt.VideoFile.str.replace('.mp4','')
        out_dt = pd.merge(c_dt, pred_dt[['ClipName','Prediction','Confidence']], on='ClipName', how = 'left')
        out_dt['modelID'] = self.fileManager.vModelID
        out_dt.to_csv(self.fileManager.localAllLabeledClustersFile)
