
import subprocess, os, sys
import pdb, datetime, os, subprocess, argparse, random, cv2
import pandas as pd
import shutil

class ManualLabelVideoFixer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, category, analysisID = None):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.category = category
		self.analysisID = analysisID
		# 10 categories of annotation plus quit and skip commands
		self.commands = ['c','f','p','t','b','m','s','x','o','d','q','k','r']
		self.commands_help = "Type 'c': BuildScoop; 'f': FeedScoop; 'p': BuildSpit; 't': FeedSpit; 'b': BuildMultiple; 'm': FeedMultiple; s': Spawn; 'x': Reflection; 'o': FishOther; 'd': DropSand; 'q': quit; 'k': skip; 'r': redo"

		assert self.category in self.commands

	def downloadProjectData(self):
		
		self.fileManager.downloadData(self.fileManager.localLabeledClipsFile)

		dt = pd.read_csv(self.fileManager.localLabeledClipsFile, index_col = 'LID')
		dt = dt[(dt.AnalysisID == self.analysisID) & (dt.ManualLabel == self.category) & (dt.OldLabel != dt.OldLabel)]
		dt['ProjectID'] = dt.ClipName.str.split('__').str[0]
		self.projectIDs = dt.ProjectID.unique().tolist()
		self.labeled_dt = dt
		for pid in self.projectIDs:
			self.fileManager.setProjectID(pid)
			#self.fileManager.downloadData(self.fileManager.localLabeledClipsProjectDir, tarred = True)

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localLabeledClipsFile)
		for pid in self.projectIDs:
			assert os.path.exists(self.fileManager.localLabeledClipsProjectDir)

	def uploadProjectData(self, delete = False):

		self.fileManager.uploadData(self.fileManager.localLabeledClipsFile)
				
		if delete:
			shutil.rmtree(self.localObjectDetectionDir)

	def fixVideos(self, initials):
		
		self.initials = initials

		# Read in annotations and create csv file for all annotations with the same user and projectID

		numClips = len(self.labeled_dt)
		# Identify clips that can be labeled
		clips = [self.fileManager.localLabeledClipsDir + row.ProjectID + '/' + row.ClipName + '.mp4' for i, row in self.labeled_dt.iterrows()]
		pdb.set_trace()
		for index, row in self.labeled_dt.iterrows()

		print(self.commands_help)
		
		random.shuffle(clips) # Shuffle the clips so that it's a random sample

		index = 0
		while index < len(clips): # We use a while loop so we can reannotate a clip if a mistake is made
			f = clips[index] # Get current clip
			clip_name = self.fileManager.projectID + '__' + f.split('/')[-1].replace('_ManualLabel.mp4','')
			if clip_name in labeled_dt.ClipName.values:
				print('Skipping ' + clip_name + ' since it is already labeled', file = sys.stderr)
				index += 1
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

			if clip_name in labeled_dt.ClipName.values:
				labeled_dt.loc[newlyLabeled_dt.ClipName == clip_name,'ManualLabel'] = chr(info)
			else:
				labeled_dt.loc[len(labeled_dt)] = [self.fileManager.analysisID, clip_name, chr(info), self.initials, str(datetime.datetime.now())] # Create new annotation

			labeled_dt.to_csv(self.fileManager.localLabeledClipsFile, sep = ',')

			# subprocess.run(['mv', self.fileManager.localManualLabelClipsDir + f.replace('_ManualLabel',''), self.fileManager.localNewLabeledClipsDir])
			shutil.move(f.replace('_ManualLabel',''), self.fileManager.localLabeledClipsProjectDir + clip_name + '.mp4') #changed for windows
			annotatedClips += 1
			index += 1

			if annotatedClips >= self.number:
				break

		return annotatedClips

	
			