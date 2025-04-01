
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

	def __init__(self, fileManager, initials, number):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.initials = initials
		self.number = number

		# 10 categories of annotation plus quit and skip commands
		self.commands = ['c','f','p','t','b','m','s','x','o','d','q','k','r']
		self.commands_help = "Type 'c': BuildScoop; 'f': FeedScoop; 'p': BuildSpit; 't': FeedSpit; 'b': BuildMultiple; 'm': FeedMultiple; s': Spawn; 'x': Reflection; 'o': FishOther; 'd': DropSand; 'q': quit; 'k': skip; 'r': redo"

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManger.localMasterDir)
        self.fileManager.createDirectory(self.fileManger.localAnalysisDir)
        self.fileManager.createDirectory(self.fileManger.localNewLabeledClipsDir)
        self.fileManger.downloadData(self.fileManger.localManualLabelClipsDir, tarred_subdirs = True)
        self.fileManager.downloadData(self.fileManger.localLabeledClipsFile)

	def validateInputData(self):

		assert os.path.exists(self.fileManager.localManualLabelClipsDir)
		assert os.path.exists(self.fileManager.localNewLabeledClipsDir)
		assert os.path.exists(self.fileManager.localLabeledClipsFile)
	
	def uploadProjectData(self, delete = False):
        self.uploadAndMerge(self.localNewLabeledVideosFile, self.localLabeledClipsFile, ID = 'LID')
        self.uploadAndMerge(self.localNewLabeledClipsDir, self.localLabeledClipsProjectDir, tarred = True)
                # pdb.set_trace()

        if delete:
            shutil.rmtree(self.localProjectDir)


	def labelVideos(self):

		# Read in annotations and create csv file for all annotations with the same user and projectID
		previouslyLabeled_dt = pd.read_csv(self.fileManager.localLabeledClipsFile, index_col = 'LID')
		newlyLabeled_dt = pd.DataFrame(columns =previouslyLabeled_dt.columns)
		newlyLabeled_dt.index.name = 'LID'

		# Identify clips that can be labeled
		clips = [x for x in os.listdir(self.fileManager.localManualLabelClipsDir) if 'ManualLabel.mp4' in x]

		print(self.commands_help)
		
		annotatedClips = 0 # Keep track of all the new clips that have been labeled
		random.shuffle(clips) # Shuffle the clips so that it's a random sample

		index = 0
		while index < len(clips): # We use a while loop so we can reannotate a clip if a mistake is made
			f = clips[index] # Get current clip

			if not previouslyLabeled_dt.loc[previouslyLabeled_dt.ClipName == f]['ManualLabel'].empty:
				print('Skipping ' + f + ' since it is already labeled', file = sys.stderr)
				continue
	
			cap = cv2.VideoCapture(self.fileManager.localManualLabelClipsDir + f) # Open video object and display it
	
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
				return

			if info == ord('k'):
				index += 1
				continue #skip

			if info == ord('r'):
				index = index - 1
				continue

			clip_name = self.fileManager.projectID + '__' + f.replace('_ManualLabel.mp4','')

			if clip_name in newlyLabeled_dt.ClipName:
				newlyLabeled_dt.loc[newlyLabeled_dt.ClipName == clip_name,'ManualLabel'] = chr(info)
			else:
				newlyLabeled_dt.loc[len(newlyLabeled_dt)] = [clip_name, chr(info), self.initials, str(datetime.datetime.now())] # Create new annotation

			newlyLabeled_dt.to_csv(self.fileManager.localNewLabeledVideosFile, sep = ',')

			# subprocess.run(['mv', self.fileManager.localManualLabelClipsDir + f.replace('_ManualLabel',''), self.fileManager.localNewLabeledClipsDir])
			shutil.move(self.fileManager.localManualLabelClipsDir + f.replace('_ManualLabel',''), self.fileManager.localNewLabeledClipsDir) #changed for windows
			annotatedClips += 1
			index += 1

			if annotatedClips >= self.number:
				break
		return annotatedClips
