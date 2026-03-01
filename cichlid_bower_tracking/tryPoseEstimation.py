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

		train_labels = os.listdir(subdir + 'labels/train/') 
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

		val_labels = os.listdir(subdir + 'labels/val/') 
		for ti in val_labels:
			with open(subdir + 'labels/val/' + ti) as infile, open(outdir + 'labels/val/' + ti, 'w') as outfile:
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

		val_images = os.listdir(subdir + 'images/val/') 
		val_images_filter = os.listdir(main_dir_detect + project + 'images/val/')
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
			else:
				pdb.set_trace()

	for d in [outdir_pose1,outdir_pose2,outdir_detect1,outdir_detect2]:
		fm_obj.uploadData(d)


def train_model(mode):
	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	fm_obj.downloadData(fm_obj.localPoseDir + 'Benthics/')
	model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

	if mode == 'pose':
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

#modifyTuckerObjectDetections()
modifyTuckerLabels()

#train_model('pose')
#train_model('detect')