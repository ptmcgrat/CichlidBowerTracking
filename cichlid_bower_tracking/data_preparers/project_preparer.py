import datetime, os, subprocess, pdb, math
import pandas as pd
import sys
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

class ProjectPreparer():
    # This class takes in a projectID and runs all the appropriate analysis

    def __init__(self, projectID = None, modelID = None, workers = None, analysisID=None):
        self.projectID = projectID
        if modelID == 'None':
            modelID = None
        self.fileManager = FM(projectID = projectID, modelID = modelID, analysisID=analysisID)
        self.modelID = modelID
        #if not self._checkProjectID():
        #    raise Exception(projectID + ' is not valid.')
        self.workers = workers

    def _checkProjectID(self):
        if self.projectID is None:
            return True
        projectIDs = subprocess.run(['rclone', 'lsf', self.fileManager.cloudMasterDir + '__ProjectData/'], capture_output = True, encoding = 'utf-8').stdout.split()
        if self.projectID + '/' in projectIDs:
            return True
        else:
            pdb.set_trace()
            return False

    def downloadData(self, dtype, videoIndex = None):
        self.fileManager.downloadProjectData(dtype, videoIndex)

    def uploadData(self, dtype, videoIndex = None, delete = False, no_upload = False):
        self.fileManager.uploadProjectData(dtype, videoIndex, delete, no_upload)

    def runPrepAnalysis(self):
        from cichlid_bower_tracking.data_preparers.prep_preparer import PrepPreparer as PrP
        prp_obj = PrP(self.fileManager)
        prp_obj.validateInputData()
        prp_obj.prepData()

    def runDepthAnalysis(self):
        from cichlid_bower_tracking.data_preparers.depth_preparer import DepthPreparer as DP

        dp_obj = DP(self.fileManager)
        dp_obj.validateInputData()
        dp_obj.createSmoothedArray()
        dp_obj.createDepthFigures()
        dp_obj.createRGBVideo()

    def runClusterAnalysis(self, videoIndex):
        from cichlid_bower_tracking.data_preparers.cluster_preparer import ClusterPreparer as CP

        if videoIndex is None:
            videos = list(range(len(self.fileManager.lp.movies)))
        else:
            videos = [videoIndex]
        for videoIndex in videos:
            cp_obj = CP(self.fileManager, videoIndex, self.workers)
            cp_obj.validateInputData()
            cp_obj.runClusterAnalysis()

    def runTrackFishAnalysis(self, videoIndexIn):
        
        import GPUtil
        from cichlid_bower_tracking.data_preparers.fish_tracking_preparer import FishTrackingPreparer as FTP
        if videoIndexIn is None:
            videos = list(range(len(self.fileManager.lp.movies)))
        else:
            videos = [videoIndexIn]
        
        ftp_objs = []
        for videoIndex in videos:
            ftp_objs.append(FTP(self.fileManager, videoIndex))
            ftp_objs[-1].validateInputData()

        available_cards = GPUtil.getAvailable(order = 'first', maxMemory = 0.2, limit = 8)

        current_idx = 0
        while current_idx < len(videos):
            processes = []
            for i in range(4):
                for gpu in available_cards:
                    if current_idx < len(videos):
                        processes.append(ftp_objs[current_idx].runObjectDetectionAnalysis(gpu))
                        current_idx += 1


            for p1 in processes:
                p1.communicate()
                if p1.returncode != 0:
                    print('YOLO Error')
        processes = []
        for idx in range(len(videos)):
            processes.append(ftp_objs[idx].runSORT())

        for p1 in processes:
            p1.communicate()
            if p1.returncode != 0:
                print('SORT Error')

        # Combine predictions
        if videoIndexIn is None:
            for videoIndex in videos:
                videoObj = self.fileManager.returnVideoObject(videoIndex)
                new_dt_t = pd.read_csv(videoObj.localFishTracksFile)
                new_dt_d = pd.read_csv(videoObj.localFishDetectionsFile)
                try:
                    c_dt_t = c_dt_t.append(new_dt_t)
                    c_dt_d = c_dt_d.append(new_dt_d)

                except NameError:
                    c_dt_t = new_dt_t
                    c_dt_d = new_dt_d

            c_dt_t.to_csv(self.fileManager.localAllFishTracksFile)
            c_dt_d.to_csv(self.fileManager.localAllFishDetectionsFile)


    def runClusterTrackAssociationAnalysis(self):
        from cichlid_bower_tracking.data_preparers.cluster_track_association_preparer import ClusterTrackAssociationPreparer as CTAP
        ctap_obj = CTAP(self.fileManager)
        ctap_obj.validateInputData()
        ctap_obj.runAssociationAnalysis()

    def run3DClassification(self):
        from cichlid_bower_tracking.data_preparers.threeD_classifier_preparer import ThreeDClassifierPreparer as TDCP

        tdcp_obj = TDCP(self.fileManager)
        tdcp_obj.validateInputData()
        tdcp_obj.predictLabels()
        tdcp_obj.createSummaryFile()

    def manuallyLabelVideos(self, initials, number):
        from cichlid_bower_tracking.data_preparers.manual_label_video_preparer import ManualLabelVideoPreparer as MLVP
        mlv_obj = MLVP(self.fileManager, initials, number)
        mlv_obj.validateInputData()
        mlv_obj.labelVideos()


    def createModel(self, MLtype, projectIDs, gpu):
        from cichlid_bower_tracking.data_preparers.threeD_model_preparer import ThreeDModelPreparer as TDMP

        if MLtype == '3DResnet':
            tdm_obj = TDMP(self.fileManager, projectIDs, self.modelID, gpu)
            tdm_obj.validateInputData()
            tdm_obj.create3DModel()

    def runMLFishDetection(self, videoIndexIn):
        from cichlid_bower_tracking.data_preparers.add_fish_sex_preparer import  AddFishSexPreparer as AFSP
        if videoIndexIn is None:
            videos = list(range(len(self.fileManager.lp.movies)))
        else:
            videos = [videoIndexIn]
        
        AFSP(self.fileManager, videoIndexIn)
        # Combine predictions
        if videoIndexIn is None:
            for videoIndex in videos:
                videoObj = self.fileManager.returnVideoObject(videoIndex)
                new_dt_s = pd.read_csv(videoObj.localFishSexFile)
                try:
                    c_dt_s = c_dt_s.append(new_dt_s)
                except NameError:
                    c_dt_s = new_dt_s
        else:
            c_dt_s = pd.read_csv(videoObj.localFishSexFile)
                    
            c_dt_s.to_csv(self.fileManager.localAllFishSexFile)
            
    def runSummaryCreation(self):
        from cichlid_bower_tracking.data_preparers.summary_preparer import SummaryPreparer as SP
        sp_obj = SP(self.fileManager)
        sp_obj.createFullSummary()

    def backupAnalysis(self):
        uploadCommands = set()

        uploadFiles = [x for x in os.listdir(self.fileManager.localUploadDir) if 'UploadData' in x]

        for uFile in uploadFiles:
            with open(self.fileManager.localUploadDir + uFile) as f:
                line = next(f)
                for line in f:
                    tokens = line.rstrip().split(',')
                    tokens[2] = bool(int(tokens[2]))
                    uploadCommands.add(tuple(tokens))

        for command in uploadCommands:
            self.fileManager.uploadData(command[0], command[1], command[2])

        for uFile in uploadFiles:
            subprocess.run(['rm', '-rf', self.fileManager.localUploadDir + uFile])

        self.fileManager.uploadData(self.fileManager.localAnalysisLogDir, self.fileManager.cloudAnalysisLogDir, False)
        subprocess.run(['rm', '-rf', self.projFileManager.localMasterDir])

    def localDelete(self):
        subprocess.run(['rm', '-rf', self.fileManager.localProjectDir])

    def createUploadFile(self, uploads):
        with open(self.fileManager.localUploadDir + 'UploadData_' + str(datetime.datetime.now().timestamp()) + '.csv', 'w') as f:
            print('Local,Cloud,Tar', file = f)
            for upload in uploads:
                print(upload[0] + ',' + upload[1] + ',' + str(upload[2]), file = f)

    def createAnalysisUpdate(self, aType, procObj):
        now = datetime.datetime.now()
        with open(self.fileManager.localAnalysisLogDir + 'AnalysisUpdate_' + str(now.timestamp()) + '.csv', 'w') as f:
            print('ProjectID,Type,Version,Date', file = f)
            print(self.projectID + ',' + aType + ',' + procObj.__version__ + '_' + os.getenv('USER') + ',' + str(now), file= f)
