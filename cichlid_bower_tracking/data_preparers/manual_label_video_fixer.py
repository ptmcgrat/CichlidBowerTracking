
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

	def __init__(self, fileManager, category, analysisID, projectIDs):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.category = category
		self.analysisID = analysisID
		self.projectIDs = projectIDs
		# 10 categories of annotation plus quit and skip commands
		self.commands = ['c','f','p','t','b','m','s','x','o','d','q','k','r']
		self.commands_help = "Type 'c': BuildScoop; 'f': FeedScoop; 'p': BuildSpit; 't': FeedSpit; 'b': BuildMultiple; 'm': FeedMultiple; s': Spawn; 'x': Reflection; 'o': FishOther; 'd': DropSand; 'q': quit; 'k': skip; 'r': redo"

		assert self.category in self.commands

	def downloadData(self):
		
		self.fileManager.downloadData(self.fileManager.localLabeledClipsFile)

		for pid in self.projectIDs:
			self.fileManager.setProjectID(pid)
			self.fileManager.downloadData(self.fileManager.localLabeledClipsProjectDir, tarred = True)

	def validateInputData(self):
		assert os.path.exists(self.fileManager.localLabeledClipsFile)
		for pid in self.projectIDs:
			assert os.path.exists(self.fileManager.localLabeledClipsProjectDir)

	def uploadData(self, delete = False):

		self.fileManager.uploadData(self.fileManager.localLabeledClipsFile)
				
		if delete:
			shutil.rmtree(self.fileManager.local3DVideosDir)

	def fixVideos(self, initials):
		
		o_dt = pd.read_csv(self.fileManager.localLabeledClipsFile, index_col = 'LID')
		dt = o_dt[(o_dt.AnalysisID == self.analysisID) & (o_dt.ManualLabel == self.category) & (o_dt.OldLabel != o_dt.OldLabel)]
		# Read in annotations and create csv file for all annotations with the same user and projectID

		# Identify clips that can be labeled
		clips = []
		for projectID in self.projectIDs:
			sub_dt = dt[dt.ProjectID == projectID]
			self.fileManager.setProjectID(projectID)
			clips += [(x,self.fileManager.localLabeledClipsProjectDir + x + '_ManualLabel.mp4') for x in sub_dt.ClipName.tolist()]

		assert all(os.path.exists(x[1]) for x in clips)

		print(self.commands_help)
		
		index = 0
		while index < len(clips): # We use a while loop so we can reannotate a clip if a mistake is made
			clip_name, f = clips[index] # Get current clip
			
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
				return 

			if info == ord('k'):
				index += 1
				continue #skip

			if info == ord('r'):
				index = index - 1
				continue

			row = o_dt.loc[o_dt.ClipName == clip_name].iloc[0]
			o_dt.loc[o_dt.ClipName == clip_name,'OldLabel'] = row.ManualLabel + '_' + row.MLabeler + '_' + row.MLabelTime
			o_dt.loc[o_dt.ClipName == clip_name,'ManualLabel'] = chr(info)
			o_dt.loc[o_dt.ClipName == clip_name,'MLabeler'] = initials
			o_dt.loc[o_dt.ClipName == clip_name,'MLabelTime'] = str(datetime.datetime.now())

			o_dt.to_csv(self.fileManager.localLabeledClipsFile, sep = ',')

			index += 1


	
			