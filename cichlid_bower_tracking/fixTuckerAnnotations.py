import subprocess,pdb,shutil
from helper_modules.file_manager import FileManager as FM

fm_obj = FM()
fm_obj.downloadData(fm_obj.localYOLOAnnotationDir + 'OriginalTuckerData/')

hybrid_indata = fm_obj.localYOLOAnnotationDir + 'CVxMC-tucker-2025-11-13/'
hybrid_outdata = fm_obj.localYOLOAnnotationDir + 'HybridMulti/'
fm_obj.createDirectory(hybrid_outdata)
fm_obj.createDirectory(hybrid_outdata + 'images/train/')
fm_obj.createDirectory(hybrid_outdata + 'images/val/')
fm_obj.createDirectory(hybrid_outdata + 'labels/train/')
fm_obj.createDirectory(hybrid_outdata + 'labels/val/')
fm_obj.createDirectory(hybrid_outdata)

with open(hybrid_outdata + 'data.yaml', 'w') as f:
	print('path: ' + hybrid_outdata)
	print('train: images/train')
	print('val: images/val')
	print('')
	print('names:')
	print('  0: female')
	print('  1: male')

train_images = os.listdir(hybrid_indata + 'images/train/') 
for ti in train_images:
	subprocess.run(['cp', hybrid_indata + 'images/train/' + ti, hybrid_outdata + 'images/train/' + ti])
val_images = os.listdir(hybrid_indata + 'images/val/') 
for ti in val_images:
	subprocess.run(['cp', hybrid_indata + 'images/train/' + ti, hybrid_outdata + 'images/train/' + ti])

train_labels = os.listdir(hybrid_indata + 'labels/train/') 
for ti in train_labels:
	with open(hybrid_indata + 'labels/train/' + ti) as infile, open(hybrid_outdata + 'labels/train' + ti) as outfile:
		for line in infile:
			if line[0] == '0':
				print('1' + line[1:], file = outfile)
			elif line[0] == '1':
				print('0' + line[1:], file = outfile)
			else:
				raise Exception

val_labels = os.listdir(hybrid_indata + 'labels/val/') 
for ti in val_labels:
	with open(hybrid_indata + 'labels/val/' + ti) as infile, open(hybrid_outdata + 'labels/val' + ti) as outfile:
		for line in infile:
			if line[0] == '0':
				print('1' + line[1:], file = outfile)
			elif line[0] == '1':
				print('0' + line[1:], file = outfile)
			else:
				raise Exception

