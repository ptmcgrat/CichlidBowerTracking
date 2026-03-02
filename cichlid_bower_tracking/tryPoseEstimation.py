import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO
import torch

def modifyTuckerLabels():
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	main_dir_pose = fm_obj.localAnnotationDir + 'TuckerAnnotations/bbox_and_pose/'
	main_dir_detect = fm_obj.localAnnotationDir + 'TuckerAnnotations/bbbox_only/OriginalTuckerData/'
	fm_obj.downloadData(main_dir_pose)
	fm_obj.downloadData(main_dir_detect)

	projects = ['CVxMC-tucker-2025-11-13/', 'MC-tucker-2025-11-03/', 'MCxYH-tucker-2025-11-03/', 'YH-tucker-2025-11-03/']
	#for dtype,outdir in zip(['bbbox_only/OriginalTuckerData/','bbox_and_pose/'],[fm_obj.localObjectDetectionDir + 'GenericSex/', fm_obj.localPoseDir + 'GenericPose/']):

	outdir_pose1 = fm_obj.localPoseDir + 'GenericPose1/'
	outdir_pose2 = fm_obj.localPoseDir + 'GenericPose2/'
	outdir_detect1 = fm_obj.localObjectDetectionDir + 'GenericSex1/'
	outdir_detect2 = fm_obj.localObjectDetectionDir + 'GenericSex2/'
	
	for d in [outdir_pose1,outdir_pose2,outdir_detect1,outdir_detect2]:
		fm_obj.createDirectory(d)

		for sd in ['images/train','images/val','labels/train','labels/val']:
			fm_obj.createDirectory(d + sd)

	for d in [outdir_pose1,outdir_pose2]:
		with open(d + 'data.yaml', 'w') as f:
			print('path: ' + d, file = f)
			print('train: images/train', file = f)
			print('val: images/val', file = f)
			print('', file = f)
			print('kpt_shape: [10, 3]', file = f)
			print('flip_idx: [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]', file = f)
			print('kpt_names:', file = f)
			print('  - nose', file = f)
			print('  - leftEye', file = f)
			print('  - rightEye', file = f)
			print('  - head', file = f)
			print('  - spine1', file = f)
			print('  - spine2', file = f)
			print('  - spine3', file = f)
			print('  - spine4', file = f)
			print('  - peduncle', file = f)
			print('  - tailTip', file = f)
			print('', file = f)
			print('names:', file = f)
			print('  0: female', file = f)
			print('  1: male', file = f)

	for d in [outdir_detect1,outdir_detect2]:
		with open(d + 'data.yaml', 'w') as f:
			print('path: ' + d, file = f)
			print('train: images/train', file = f)
			print('val: images/val', file = f)
			print('', file = f)
			print('names:', file = f)
			print('  0: female', file = f)
			print('  1: male', file = f)

	for project in projects:
		subdir = main_dir_pose + project
		subdir_filt = main_dir_detect + project
		
		train_images = os.listdir(subdir + 'images/train/') 
		train_images_filter = os.listdir(main_dir_detect + project + 'images/train/')
		val_images_filter = os.listdir(main_dir_detect + project + 'images/val/')

		for ti in train_images:
			subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_pose1 + 'images/train/' + ti])
			subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_detect1 + 'images/train/' + ti])
			ti_l = ti.replace('.png','.txt')
			with open(subdir + 'labels/train/' + ti_l) as infile, open(outdir_detect1 + 'labels/train/' + ti_l, 'w') as outfile_detect, open(outdir_pose1 + 'labels/train/' + ti_l, 'w') as outfile_pose:
				for line in infile:
					line = line.rstrip()
					tokens = line.split(' ')

					if line[0] == '0':
						print('1' + line[1:], file = outfile_pose)
						print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile_pose)
						print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)

					else:
						raise Exception

			if ti in train_images_filter:
				subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_pose2 + 'images/train/' + ti])
				subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_detect2 + 'images/train/' + ti])

				with open(subdir + 'labels/train/' + ti_l) as infile, open(outdir_detect2 + 'labels/train/' + ti_l, 'w') as outfile_detect, open(outdir_pose2 + 'labels/train/' + ti_l, 'w') as outfile_pose:
					for line in infile:
						line = line.rstrip()
						tokens = line.split(' ')

						if line[0] == '0':
							print('1' + line[1:], file = outfile_pose)
							print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						elif line[0] == '1':
							print('0' + line[1:], file = outfile_pose)
							print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)

						else:
							raise Exception

			if ti in val_images_filter:
				subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_pose2 + 'images/val/' + ti])
				subprocess.run(['cp', subdir + 'images/train/' + ti, outdir_detect2 + 'images/val/' + ti])

				with open(subdir + 'labels/train/' + ti_l) as infile, open(outdir_detect2 + 'labels/val/' + ti_l, 'w') as outfile_detect, open(outdir_pose2 + 'labels/val/' + ti_l, 'w') as outfile_pose:
					for line in infile:
						line = line.rstrip()
						tokens = line.split(' ')

						if line[0] == '0':
							print('1' + line[1:], file = outfile_pose)
							print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						elif line[0] == '1':
							print('0' + line[1:], file = outfile_pose)
							print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)

						else:
							raise Exception


		val_images = os.listdir(subdir + 'images/val/') 
		for vi in val_images:
			subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_pose1 + 'images/val/' + vi])
			subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_detect1 + 'images/val/' + vi])
			vi_l = vi.replace('.png','.txt')

			with open(subdir + 'labels/val/' + vi_l) as infile, open(outdir_detect1 + 'labels/val/' + vi_l, 'w') as outfile_detect, open(outdir_pose1 + 'labels/val/' + vi_l, 'w') as outfile_pose:
				for line in infile:
					line = line.rstrip()
					tokens = line.split(' ')

					if line[0] == '0':
						print('1' + line[1:], file = outfile_pose)
						print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile_pose)
						print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)

					else:
						raise Exception

			if vi in val_images_filter:
				subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_pose2 + 'images/val/' + vi])
				subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_detect2 + 'images/val/' + vi])

				with open(subdir + 'labels/val/' + vi_l) as infile, open(outdir_detect2 + 'labels/val/' + vi_l, 'w') as outfile_detect, open(outdir_pose2 + 'labels/val/' + vi_l, 'w') as outfile_pose:
					for line in infile:
						line = line.rstrip()
						tokens = line.split(' ')

						if line[0] == '0':
							print('1' + line[1:], file = outfile_pose)
							print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						elif line[0] == '1':
							print('0' + line[1:], file = outfile_pose)
							print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						else:
							raise Exception
			
			if vi in train_images_filter:
				subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_pose2 + 'images/train/' + vi])
				subprocess.run(['cp', subdir + 'images/val/' + vi, outdir_detect2 + 'images/train/' + vi])

				with open(subdir + 'labels/val/' + vi_l) as infile, open(outdir_detect2 + 'labels/train/' + vi_l, 'w') as outfile_detect, open(outdir_pose2 + 'labels/train/' + vi_l, 'w') as outfile_pose:
					for line in infile:
						line = line.rstrip()
						tokens = line.split(' ')

						if line[0] == '0':
							print('1' + line[1:], file = outfile_pose)
							print('1 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						elif line[0] == '1':
							print('0' + line[1:], file = outfile_pose)
							print('0 ' + ' '.join(tokens[1:5]), file = outfile_detect)
						else:
							raise Exception

	for d in [outdir_pose1,outdir_pose2,outdir_detect1,outdir_detect2]:
		fm_obj.uploadData(d)

def train_models():
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	fm_obj.downloadData(fm_obj.localPoseDir + 'GenericPose1/')
	#fm_obj.downloadData(fm_obj.localPoseDir + 'GenericPose2/')
	#fm_obj.downloadData(fm_obj.localObjectDetectionDir + 'GenericSex1/')
	#fm_obj.downloadData(fm_obj.localObjectDetectionDir + 'GenericSex2/')

	model = YOLO("yolo26x-pose.pt")  # load a pretrained model (recommended for training)
	results1 = model.train(data=fm_obj.localPoseDir + 'GenericPose1/data.yaml', epochs=100, imgsz=640, project = fm_obj.localMLPoseDir, name = 'GenericPose1', batch = -1, exist_ok=True)
	#model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
	#results2 = model.train(data=fm_obj.localPoseDir + 'GenericPose2/data.yaml', epochs=100, imgsz=640, project = fm_obj.localMLPoseDir, name = 'GenericPose2', batch = -1, exist_ok=True)
	
	#model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
	#results3 = model.train(data=fm_obj.localObjectDetectionDir + 'GenericSex1/data.yaml', epochs=100, imgsz=640, project = fm_obj.localYOLODir, name = 'GenericSex1', batch = -1, exist_ok=True)
	#model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
	#results4 = model.train(data=fm_obj.localObjectDetectionDir + 'GenericSex2/data.yaml', epochs=100, imgsz=640, project = fm_obj.localYOLODir, name = 'GenericSex2', batch = -1, exist_ok=True)
	fm_obj.uploadData(fm_obj.localMLPoseDir + 'GenericPose1/')
	#fm_obj.uploadData(fm_obj.localMLPoseDir + 'GenericPose2/')
	#fm_obj.uploadData(fm_obj.localYOLODir + 'GenericSex1/')
	#fm_obj.uploadData(fm_obj.localYOLODir + 'GenericSex2/')

def trackData():
	# Check for MPS
	#device = 'cpu' if torch.backends.mps.is_available() else 'cpu'

	fm_obj = FM(analysisID = 'YH_MC_Parentals')

	models = [fm_obj.localYOLODir + 'GenericSex1/', fm_obj.localYOLODir + 'GenericSex2/']
	models += [fm_obj.localMLPoseDir + 'GenericPose1/',fm_obj.localMLPoseDir + 'GenericPose2/']

	test_file = fm_obj.localLabeledDLCClipsDir + 'MCYHF1_549_t011_tr1__0009_vid__DLC.mp4'
	fm_obj.downloadData(test_file)

	model_dir = models[2]

	fm_obj.downloadData(model_dir)	
	model = YOLO(model_dir + 'weights/best.pt')
	results = model.track(test_file, stream = True, save = True, agnostic_nms = True)
	with open('TestTracking.csv', 'w', newline='') as f:

		writer = csv.writer(f)
		writer.writerow(['Frame','TrackID','X_c','Y_c','Width','Height','ClassID','Sex','Pose_Nose','Pose_LeftEye','Pose_RightEye', 'Pose_Head','Pose_Spine1','Pose_Spine2','Pose_Spine3','Pose_Spine4','Pose_Peduncle','Pose_TailTip'])
		for frame_idx, result in enumerate(results):

			if result.boxes.id is not None:
				boxes = result.boxes.xywh.cpu().numpy()  # Convert to numpy for easy manipulation
				track_ids = result.boxes.id.cpu().numpy().astype(int)
				classes = result.boxes.cls.cpu().numpy()
				poses = result.keypoints.xy.cpu().numpy()
				for box, track_id, class_id, pose in zip(boxes, track_ids, classes, poses):
					x_center, y_center, width, height = box
					# Write frame, ID, and coordinates to the CSV file
					writer.writerow([frame_idx, track_id, x_center, y_center, width, height, class_id, result.names[class_id]] + [(x[0],x[1]) for x in pose])

	subprocess.run(['ffmpeg','-i','runs/pose/track/MCYHF1_549_t011_tr1__0009_vid__DLC.avi','-c:v','libx264','-c:a','aac','-crf','18','-b:a','224k','runs/pose/track/MCYHF1_549_t011_tr1__0009_vid__DLC.mp4'])
	subprocess.run(['rclone','copy','runs/pose/track/MCYHF1_549_t011_tr1__0009_vid__DLC.mp4','ptm_dropbox:/CoS/BioSci/BioSci-McGrath/'])
#modifyTuckerObjectDetections()
#modifyTuckerLabels()

train_models()
trackData()