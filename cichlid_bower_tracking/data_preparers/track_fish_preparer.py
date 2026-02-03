import pdb, subprocess, os, csv
from ultralytics import YOLO

class TrackFishPreparer():
	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndex):

		self.__version__ = '1.0.0'

		self.fileManager = fileManager
		self.videoObj = self.fileManager.returnVideoObject(videoIndex)
		self.videoIndex = videoIndex

	def downloadProjectData(self):
		self.fileManager.createDirectory(self.fileManager.localMasterDir)
		self.fileManager.createDirectory(self.fileManager.localTroubleshootingDir)
		self.fileManager.createDirectory(self.fileManager.localLogfileDir)
		try:
			self.fileManager.downloadData(self.videoObj.localVideoFile)
		except FileNotFoundError:
			print(self.videoObj.localVideoFile + ' not found on cloud. Trying h264 file')
			self.fileManager.downloadData(self.videoObj.localh264File)
			command = ['ffmpeg', '-r', str(self.videoObj.framerate), '-i', self.videoObj.localh264File, '-threads', str(self.workers), '-c:v', 'copy', '-r', str(self.videoObj.framerate), self.videoObj.localVideoFile]
			ffmpeg_output = subprocess.run(command, capture_output = True)
			if ffmpeg_output.returncode != 0:
				print(ffmpeg_output.stderr.decode('utf-8'))
			assert os.path.isfile(self.videoObj.localVideoFile)
			assert os.path.getsize(self.videoObj.localVideoFile) > os.path.getsize(self.videoObj.localh264File)
			#self.fileManager.uploadData(self.videoObj.localVideoFile)
			subprocess.run(['rm', '-f', self.videoObj.localh264File])

		self.fileManager.downloadData(self.fileManagerlocalYOLOModelFile)
		self.createLogFile()

	def validateInputData(self):

		assert os.path.exists(self.videoObj.localVideoFile)
		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManagerlocalYOLOModelFile)

	def createLogFile(self):
		
		# with open(self.fileManager.localClusterLogfile,'w') as f:
		with open(self.videoObj.localYOLOLogfile,'w') as f:
			print('GitBranch: ' + self.fileManager.branch_name, file = f)
			print('Username: ' + os.getenv('USER'), file = f)
			print('Nodename: ' + os.uname().nodename, file = f)
			print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)
			output = subprocess.run(['conda','list'], capture_output = True)
			print(output.stdout.decode('utf-8'), file = f)

	def runYOLOAnalysis(self):

		model = YOLO(self.fileManagerlocalYOLOModelFile)

		with open(self.videoObj.localFishDetectionsFile, 'w', newline='') as f:
		
			writer = csv.writer(f)
			writer.writerow(['FrameNum', 'TrackID', 'X_center', 'Y_center', 'Width', 'Height', 'SexID', 'Sex'])

			results = model.track(self.videoObj.localVideoFile, stream = True, save=False, show=False, persist = True, verbose = False)  # Tracking with default tracker
			print('TrackingStart: ' + str(datetime.datetime.now()))
		
			for frame_idx, result in enumerate(results):
				if result.boxes.id is not None:
					boxes = result.boxes.xywh.cpu().numpy()  # Convert to numpy for easy manipulation
					track_ids = result.boxes.id.cpu().numpy().astype(int)
					classes = result.boxes.cls.cpu().numpy()
					for box, track_id, class_id in zip(boxes, track_ids, classes):
						x_center, y_center, width, height = box
						# Write frame, ID, and coordinates to the CSV file
						writer.writerow([frame_idx, track_id, x_center, y_center, width, height,class_id, result.names[class_id]])

		print('TrackingEnd: ' + str(datetime.datetime.now()))

	def uploadProjectData(self, delete = True):
		self.fileManager.uploadData(self.videoObj.localFishDetectionsFile)
		self.fileManager.uploadData(self.videoObj.localYOLOLogfile)

		if delete:
			shutil.rmtree(self.videoObj.localVideoFile)
			shutil.rmtree(self.videoObj.localFishDetectionsFile)
