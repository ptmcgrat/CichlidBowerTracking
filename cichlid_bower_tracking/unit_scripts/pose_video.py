import argparse, csv
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('InputVideo', type = str, help = 'Name of logfile')
parser.add_argument('OutputCSV', type = str, help = 'Name of logfile')
parser.add_argument('YOLOModel', type = str, help = 'Name of logfile')

args = parser.parse_args()


model = YOLO(args.YOLOModel)

with open(args.OutputCSV, 'w', newline='') as f:

	writer = csv.writer(f)
	writer.writerow(['FrameNum','TrackID','X_center','Y_center','Width','Height','SexID','Sex','Pose_Nose','Pose_LeftEye','Pose_RightEye', 'Pose_Head','Pose_Spine1','Pose_Spine2','Pose_Spine3','Pose_Spine4','Pose_Peduncle','Pose_TailTip'])

	results = model.track(args.InputVideo, stream = True, save=False, show=False, persist = True, verbose = False, agnostic_nms = True, tracker = 'custom_track.yaml')  # Tracking with default tracker

	for frame_idx, result in enumerate(results):
		if result.boxes.id is not None:
			boxes = result.boxes.xywh.cpu().numpy()  # Convert to numpy for easy manipulation
			track_ids = result.boxes.id.cpu().numpy().astype(int)
			classes = result.boxes.cls.cpu().numpy()
			poses = result.keypoints.xy.cpu().numpy()
			for box, track_id, class_id, pose in zip(boxes, track_ids, classes, poses):
				x_center, y_center, width, height = box
				writer.writerow([frame_idx, track_id, x_center, y_center, width, height, class_id, result.names[class_id]] + [(x[0],x[1]) for x in pose])


