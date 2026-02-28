import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO
import torch

def modifyTuckerObjectDetections():
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	indatas = ['CVxMC-tucker-2025-11-13/', 'MC-tucker-2025-11-03/', 'MCxYH-tucker-2025-11-03/', 'YH-tucker-2025-11-03/']
	#for dtype,outdir in zip(['bbbox_only/OriginalTuckerData/','bbox_and_pose/'],[fm_obj.localObjectDetectionDir + 'GenericSex/', fm_obj.localPoseDir + 'GenericPose/']):

	outdir = fm_obj.localObjectDetectionDir + 'GenericSex/'
	main_dir = fm_obj.localAnnotationDir + 'TuckerAnnotations/bbbox_only/OriginalTuckerData/'
	fm_obj.downloadData(main_dir)
	fm_obj.createDirectory(outdir)
	for d in ['images/train','images/val','labels/train','labels/val']:
		fm_obj.createDirectory(outdir + d)

	with open(outdir + 'data.yaml', 'w') as f:
		print('path: ' + outdir, file = f)
		print('train: images/train', file = f)
		print('val: images/val', file = f)
		print('', file = f)
		print('names:', file = f)
		print('  0: female', file = f)
		print('  1: male', file = f)


	for project in indatas:
		subdir = main_dir + project

		train_images = os.listdir(subdir + 'images/train/') 
		
		for ti in train_images:
			subprocess.run(['cp', subdir + 'images/train/' + ti, outdir + 'images/train/' + ti])
		
		val_images = os.listdir(subdir + 'images/val/') 
		for ti in val_images:
			subprocess.run(['cp', subdir + 'images/val/' + ti, outdir + 'images/val/' + ti])
		print(project + ',Num_train: ' + str(len(train_images)) + ',Num_val: ' + str(len(val_images)))

		train_labels = os.listdir(hybrid_indata + 'labels/train/') 
		for ti in train_labels:
			with open(subdir + 'labels/train/' + ti) as infile, open(outdir + 'labels/train/' + ti, 'w') as outfile:
				for line in infile:
					line = line.rstrip()
					if line[0] == '0':
						print('1' + line[1:], file = outfile)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile)
					else:
						raise Exception

		val_labels = os.listdir(hybrid_indata + 'labels/val/') 
		for ti in val_labels:
			with open(hybrid_indata + 'labels/val/' + ti) as infile, open(hybrid_outdata + 'labels/val/' + ti, 'w') as outfile:
				for line in infile:
					line = line.rstrip()
					if line[0] == '0':
						print('1' + line[1:], file = outfile)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile)
					else:
						raise Exception

	fm_obj.uploadData(outdir)

def modifyTuckerLabels():
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	indatas = ['CVxMC-tucker-2025-11-13/', 'MC-tucker-2025-11-03/', 'MCxYH-tucker-2025-11-03/', 'YH-tucker-2025-11-03/']
	#for dtype,outdir in zip(['bbbox_only/OriginalTuckerData/','bbox_and_pose/'],[fm_obj.localObjectDetectionDir + 'GenericSex/', fm_obj.localPoseDir + 'GenericPose/']):

	outdir = fm_obj.localPoseDir + 'GenericPose/'
	main_dir = fm_obj.localAnnotationDir + 'TuckerAnnotations/bbox_and_pose/' + dtype
	fm_obj.downloadData(main_dir)
	fm_obj.createDirectory(outdir)
	for d in ['images/train','images/val','labels/train','labels/val']:
		fm_obj.createDirectory(outdir + d)

	with open(outdir + 'data.yaml', 'w') as f:
		print('path: ' + outdir, file = f)
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

	for project in indatas:
		subdir = main_dir + project

		train_images = os.listdir(subdir + 'images/train/') 
		print(len(train_images))
		for ti in train_images:
			subprocess.run(['cp', subdir + 'images/train/' + ti, outdir + 'images/train/' + ti])
		
		val_images = os.listdir(subdir + 'images/val/') 
		for ti in val_images:
			subprocess.run(['cp', subdir + 'images/val/' + ti, outdir + 'images/val/' + ti])
		print(project + ',Num_train: ' + str(len(train_images)) + ',Num_val: ' + str(len(val_images)))

		train_labels = os.listdir(hybrid_indata + 'labels/train/') 
		for ti in train_labels:
			with open(subdir + 'labels/train/' + ti) as infile, open(outdir + 'labels/train/' + ti, 'w') as outfile:
				for line in infile:
					line = line.rstrip()
					if line[0] == '0':
						print('1' + line[1:], file = outfile)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile)
					else:
						raise Exception

		val_labels = os.listdir(hybrid_indata + 'labels/val/') 
		for ti in val_labels:
			with open(hybrid_indata + 'labels/val/' + ti) as infile, open(hybrid_outdata + 'labels/val/' + ti, 'w') as outfile:
				for line in infile:
					line = line.rstrip()
					if line[0] == '0':
						print('1' + line[1:], file = outfile)
					elif line[0] == '1':
						print('0' + line[1:], file = outfile)
					else:
						raise Exception

	fm_obj.uploadData(outdir)


def train_model(mode):
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	pdb.set_trace()
	if mode == 'pose':
		output = subprocess.run(['rclone','lsf','ptm_dropbox:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/ObjectDetection/YOLO_Annotations/bbbox_only/GenericSex/images/train/'], capture_output = True)
		output2 = subprocess.run(['rclone','lsf','ptm_dropbox:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/PoseData/Benthics/images/train/'], capture_output = True)

		fm_obj.downloadData(fm_obj.localPoseDir + 'Benthics/')
		model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

		results = model.train(data=fm_obj.localPoseDir + 'Benthics/data.yaml', epochs=100, imgsz=640, project = fm_obj.localMLPoseDir, name = 'Benthics', batch = -1, exist_ok=True)
		fm_obj.uploadData(fm_obj.localMLPoseDir + 'Benthics/')

	elif mode == 'detect':
		fm_obj.downloadData(fm_obj.localYOLOAnnotationDir + 'bbbox_only/GenericSex/')
		model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
		results = model.train(data=fm_obj.localYOLOAnnotationDir + 'GenericSex/data.yaml', epochs=100, imgsz=640, project = fm_obj.localYOLODir, name = 'GenericSexTest', batch = -1, exist_ok=True)
		fm_obj.uploadData(fm_obj.localYOLODir + 'GenericSexTest/')

def trackData(mode):
	# Check for MPS
	device = 'cpu' if torch.backends.mps.is_available() else 'cpu'

	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	test_file = fm_obj.localLabeledDLCClipsDir + 'MCYHF1_549_t011_tr1__0009_vid__DLC.mp4'
	fm_obj.downloadData(test_file)
	
	if mode == 'pose':
		fm_obj.downloadData(fm_obj.localMLPoseDir + 'Benthics/')
		#fm_obj.downloadData(fm_obj.localMLPoseDir + 'Benthics/weights/best.pt')

		model_pose = YOLO(fm_obj.localMLPoseDir + 'Benthics/weights/best.pt')
		results = model_pose.track(test_file, show=True, device=device)

	if mode == 'detect':
		fm_obj.downloadData(fm_obj.localYOLODir + 'GenericSexTest/')
		model_track = YOLO(fm_obj.localMLPoseDir + 'GenericSexTest/weights/best.pt')
		results = model_track.track(test_file, show=True, device=device)

modifyTuckerObjectDetections()
modifyTuckerLabels()

#train_model('pose')
#train_model('detect')