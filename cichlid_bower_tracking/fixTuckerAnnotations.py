import subprocess,pdb,shutil,os
from helper_modules.file_manager import FileManager as FM

fm_obj = FM()
fm_obj.downloadData(fm_obj.localYOLOAnnotationDir + 'OriginalTuckerData/')

hybrid_indata = fm_obj.localYOLOAnnotationDir + 'OriginalTuckerData/CVxMC-tucker-2025-11-13/'
hybrid_outdata = fm_obj.localYOLOAnnotationDir + 'HybridMulti/'
fm_obj.createDirectory(hybrid_outdata)
fm_obj.createDirectory(hybrid_outdata + 'images/train/')
fm_obj.createDirectory(hybrid_outdata + 'images/val/')
fm_obj.createDirectory(hybrid_outdata + 'labels/train/')
fm_obj.createDirectory(hybrid_outdata + 'labels/val/')
fm_obj.createDirectory(hybrid_outdata)

with open(hybrid_outdata + 'data.yaml', 'w') as f:
	print('path: ' + hybrid_outdata, file = f)
	print('train: images/train', file = f)
	print('val: images/val', file = f)
	print('', file = f)
	print('names:', file = f)
	print('  0: female', file = f)
	print('  1: male', file = f)

train_images = os.listdir(hybrid_indata + 'images/train/') 
for ti in train_images:
	subprocess.run(['cp', hybrid_indata + 'images/train/' + ti, hybrid_outdata + 'images/train/' + ti])
val_images = os.listdir(hybrid_indata + 'images/val/') 
for ti in val_images:
	subprocess.run(['cp', hybrid_indata + 'images/val/' + ti, hybrid_outdata + 'images/val/' + ti])

train_labels = os.listdir(hybrid_indata + 'labels/train/') 
for ti in train_labels:
	with open(hybrid_indata + 'labels/train/' + ti) as infile, open(hybrid_outdata + 'labels/train/' + ti, 'w') as outfile:
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

hybrid_outdata = fm_obj.localYOLOAnnotationDir + 'YH_MC_Parentals/'
fm_obj.createDirectory(hybrid_outdata)
fm_obj.createDirectory(hybrid_outdata + 'images/train/')
fm_obj.createDirectory(hybrid_outdata + 'images/val/')
fm_obj.createDirectory(hybrid_outdata + 'labels/train/')
fm_obj.createDirectory(hybrid_outdata + 'labels/val/')
fm_obj.createDirectory(hybrid_outdata)

with open(hybrid_outdata + 'data.yaml', 'w') as f:
	print('path: ' + hybrid_outdata, file = f)
	print('train: images/train', file = f)
	print('val: images/val', file = f)
	print('', file = f)
	print('names:', file = f)
	print('  0: female', file = f)
	print('  1: MC male', file = f)
	print('  2: Hybrid male', file = f)
	print('  3: YH male', file = f)

indatas = ['MC-tucker-2025-11-03/', 'MCxYH-tucker-2025-11-03/', 'YH-tucker-2025-11-03/']
for i, indata in enumerate(indatas):
	hybrid_indata = fm_obj.localYOLOAnnotationDir + 'OriginalTuckerData/' + indata
	for ti in train_images:
		subprocess.run(['cp', hybrid_indata + 'images/train/' + ti, hybrid_outdata + 'images/train/' + ti])
	val_images = os.listdir(hybrid_indata + 'images/val/') 
	for ti in val_images:
		subprocess.run(['cp', hybrid_indata + 'images/val/' + ti, hybrid_outdata + 'images/val/' + ti])
	train_labels = os.listdir(hybrid_indata + 'labels/train/') 
	for ti in train_labels:
		with open(hybrid_indata + 'labels/train/' + ti) as infile, open(hybrid_outdata + 'labels/train/' + ti, 'w') as outfile:
			for line in infile:
				line = line.rstrip()
				if line[0] == '0':
					print(str(i+1) + line[1:], file = outfile)
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
					print(str(i+1) + line[1:], file = outfile)
				elif line[0] == '1':
					print('0' + line[1:], file = outfile)
				else:
					raise Exception

hybrid_outdata = fm_obj.localYOLOAnnotationDir + 'GenericSex/'
fm_obj.createDirectory(hybrid_outdata)
fm_obj.createDirectory(hybrid_outdata + 'images/train/')
fm_obj.createDirectory(hybrid_outdata + 'images/val/')
fm_obj.createDirectory(hybrid_outdata + 'labels/train/')
fm_obj.createDirectory(hybrid_outdata + 'labels/val/')
fm_obj.createDirectory(hybrid_outdata)

with open(hybrid_outdata + 'data.yaml', 'w') as f:
	print('path: ' + hybrid_outdata, file = f)
	print('train: images/train', file = f)
	print('val: images/val', file = f)
	print('', file = f)
	print('names:', file = f)
	print('  0: female', file = f)
	print('  1: male', file = f)

indatas = ['CVxMC-tucker-2025-11-13/', 'MC-tucker-2025-11-03/', 'MCxYH-tucker-2025-11-03/', 'YH-tucker-2025-11-03/']
for i, indata in enumerate(indatas):
	hybrid_indata = fm_obj.localYOLOAnnotationDir + 'OriginalTuckerData/' + indata
	for ti in train_images:
		subprocess.run(['cp', hybrid_indata + 'images/train/' + ti, hybrid_outdata + 'images/train/' + ti])
	val_images = os.listdir(hybrid_indata + 'images/val/') 
	for ti in val_images:
		subprocess.run(['cp', hybrid_indata + 'images/val/' + ti, hybrid_outdata + 'images/val/' + ti])
	train_labels = os.listdir(hybrid_indata + 'labels/train/') 
	for ti in train_labels:
		with open(hybrid_indata + 'labels/train/' + ti) as infile, open(hybrid_outdata + 'labels/train/' + ti, 'w') as outfile:
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
