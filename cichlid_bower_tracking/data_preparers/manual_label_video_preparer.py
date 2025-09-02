
import subprocess, os
import pdb, datetime, os, subprocess, argparse, random, cv2
import pandas as pd
import shutil

class ManualLabelVideoPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, number, videoIndices, dtype = 'Videos'):

		self.__version__ = '1.0.0'
		self.quit = False
		self.fileManager = fileManager
		self.number = number
		self.videoIndices = videoIndices
		self.dtype = dtype
		# 10 categories of annotation plus quit and skip commands
		self.commands = ['c','f','p','t','b','m','s','x','o','d','q','k','r']
		self.commands_help = "Type 'c': BuildScoop; 'f': FeedScoop; 'p': BuildSpit; 't': FeedSpit; 'b': BuildMultiple; 'm': FeedMultiple; s': Spawn; 'x': Reflection; 'o': FishOther; 'd': DropSand; 'q': quit; 'k': skip; 'r': redo"

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
		
		if self.dtype == 'Videos':
			if self.fileManager.checkFileExists(self.fileManager.localLabeledClipsProjectDir + '.tar'):
				self.fileManger.downloadData(self.fileManager.localLabeledClipsProjectDir, tarred = True)
			else:
				self.fileManager.createDirectory(self.fileManager.localLabeledClipsProjectDir)
			self.fileManager.downloadData(self.fileManager.localLabeledClipsFile)
			for videoIndex in self.videoIndices:
				videoObj = self.fileManager.returnVideoObject(videoIndex)
				self.fileManager.downloadData(videoObj.localManualLabelClipsDir, tarred = True)
		if self.dtype == 'Frames':
			if not os.path.exists(self.fileManager.localLabeledFramesDir):
				if self.fileManager.checkFileExists(self.fileManager.localLabeledFramesDir + '.tar'):
					self.fileManger.downloadData(self.fileManager.localLabeledFramesDir, tarred = True)
				else:
					self.fileManager.createDirectory(self.fileManager.localLabeledFramesDir)
			self.fileManager.downloadData(self.fileManager.localLabeledFramesFile)
			for videoIndex in self.videoIndices:
				videoObj = self.fileManager.returnVideoObject(videoIndex)
				self.fileManager.downloadData(videoObj.localManualLabelFramesDir, tarred = True)

	def validateInputData(self):
		if self.dtype == 'Videos':
			assert os.path.exists(self.fileManager.localLabeledClipsProjectDir)
			assert os.path.exists(self.fileManager.localLabeledClipsFile)
			for videoIndex in self.videoIndices:
				videoObj = self.fileManager.returnVideoObject(videoIndex)
				assert os.path.exists(videoObj.localManualLabelClipsDir)
		if self.dtype == 'Frames':
			assert os.path.exists(self.fileManager.localLabeledFramesDir)
			assert os.path.exists(self.fileManager.localLabeledFramesFile)
			for videoIndex in self.videoIndices:
				videoObj = self.fileManager.returnVideoObject(videoIndex)
				assert os.path.exists(videoObj.localManualLabelFramesDir)

	
	def uploadProjectData(self, delete = False, full_delete = False, just_delete = False):

		if self.dtype == 'Videos':

			self.fileManager.uploadData(self.fileManager.localLabeledClipsProjectDir, tarred = True)
			self.fileManager.uploadData(self.fileManager.localLabeledClipsFile)

		if self.dtype == 'Frames':
			if not just_delete:
				self.fileManager.uploadData(self.fileManager.localLabeledFramesDir, tarred = True)
				self.fileManager.uploadData(self.fileManager.localLabeledFramesFile)

		if delete or just_delete:
			shutil.rmtree(self.fileManager.localProjectDir)
			shutil.rmtree(self.fileManager.localLabeledClipsProjectDir)

		if full_delete:
			shutil.rmtree(self.fileManager.localLabeledFramesDir)
			shutil.rmtree(self.fileManager.localLabeledFramesFile)

		return self.quit

	def labelVideos(self, initials):
		
		self.initials = initials

		# Read in annotations and create csv file for all annotations with the same user and projectID
		labeled_dt = pd.read_csv(self.fileManager.localLabeledClipsFile, index_col = 'LID')
		
		projectIDs = labeled_dt['ClipName'].str.split('__').str[0]
		annotatedClips = projectIDs[projectIDs == self.fileManager.projectID].shape[0]
		# Identify clips that can be labeled
		clips = []
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			clips += [videoObj.localManualLabelClipsDir + x for x in os.listdir(videoObj.localManualLabelClipsDir) if 'ManualLabel.mp4' in x]

		print(self.commands_help)
		
		random.shuffle(clips) # Shuffle the clips so that it's a random sample

		index = 0
		while index < len(clips): # We use a while loop so we can reannotate a clip if a mistake is made
			f = clips[index] # Get current clip
			clip_name = self.fileManager.projectID + '__' + f.split('/')[-1].replace('_ManualLabel.mp4','')
			if clip_name in labeled_dt.ClipName:
				print('Skipping ' + clip_name + ' since it is already labeled', file = sys.stderr)
				continue
	
			cap = cv2.VideoCapture(f) # Open video object and display it
	
			while(True):
				ret, frame = cap.read()
				if not ret:
					cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
					continue
				cv2.imshow(self.commands_help,cv2.resize(frame,(0,0),fx=4, fy=4))
				info = cv2.waitKey(25)
	
				if info in [ord(x) for x in self.commands]:
					for i in range(1,10):
						cv2.destroyAllWindows()
						cv2.waitKey(1)
					break

			if info == ord('q'):
				self.quit = True
				return annotatedClips

			if info == ord('k'):
				index += 1
				continue #skip

			if info == ord('r'):
				index = index - 1
				continue

			if clip_name in labeled_dt.ClipName:
				labeled_dt.loc[newlyLabeled_dt.ClipName == clip_name,'ManualLabel'] = chr(info)
			else:
				labeled_dt.loc[len(labeled_dt)] = [clip_name, chr(info), self.initials, str(datetime.datetime.now())] # Create new annotation

			labeled_dt.to_csv(self.fileManager.localLabeledClipsFile, sep = ',')

			# subprocess.run(['mv', self.fileManager.localManualLabelClipsDir + f.replace('_ManualLabel',''), self.fileManager.localNewLabeledClipsDir])
			shutil.move(f.replace('_ManualLabel',''), self.fileManager.localLabeledClipsProjectDir + clip_name + '.mp4') #changed for windows
			annotatedClips += 1
			index += 1

			if annotatedClips >= self.number:
				break

		return annotatedClips

	def sortFrames(self):
		# Read in annotations and create csv file for all annotations with the same user and projectID
		labeled_dt = pd.read_csv(self.fileManager.localLabeledFramesFile, index_col = 'FID')
		annotatedFrames = labeled_dt[labeled_dt.ProjectID == self.fileManager.projectID].shape[0]

		frames = []
		for videoIndex in self.videoIndices:
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			frames += [videoObj.localManualLabelFramesDir + x for x in os.listdir(videoObj.localManualLabelFramesDir)]

		while index < len(clips): # We use a while loop so we can reannotate a clip if a mistake is made
			f = clips[index] # Get current clip
			frames_name = self.fileManager.projectID + '__' + f.split('/')[-1]
			if clip_name in labeled_dt.FrameName:
				print('Skipping ' + clip_name + ' since it is already labeled', file = sys.stderr)
				continue

			labeled_dt.loc[len(labeled_dt)] = [self.fileManager.analysisID, self.fileManager.projectID, frame_name, '', ''] # Create new annotation
			shutil.move(f, self.fileManager.localLabeledFramesDir + clip_name) #changed for windows
			annotatedFrames += 1
			index += 1
			
			labeled_dt.to_csv(self.fileManager.localLabeledFramesFile, sep = ',')

			if annotatedClips >= self.number:
				break
		return annotatedFrames
