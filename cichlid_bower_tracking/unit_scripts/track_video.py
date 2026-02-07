import argparse, csv
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('InputVideo', type = str, help = 'Name of logfile')
parser.add_argument('OutputCSV', type = str, help = 'Name of logfile')
parser.add_argument('YOLOModel', type = str, help = 'Name of logfile')
parser.add_argument('AnalysisID', type = str, help = 'Name of logfile')
parser.add_argument('ProjectID', type = str, help = 'Name of logfile')


args = parser.parse_args()


model = YOLO(self.fileManager.localYOLOModelFile)

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
