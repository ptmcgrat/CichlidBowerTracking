import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO
import torch

def train_model(mode):
	fm_obj = FM(analysisID = 'YH_MC_Parentals')

	if mode == 'pose':

		fm_obj.downloadData(fm_obj.localPoseDir + 'Benthics/')
		model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

		results = model.train(data=fm_obj.localPoseDir + 'Benthics/data.yaml', epochs=100, imgsz=640, project = fm_obj.localMLPoseDir, name = 'Benthics', batch = -1, exist_ok=True)
		fm_obj.uploadData(fm_obj.localMLPoseDir + 'Benthics/')

	elif mode == 'detect':
		fm_obj.downloadData(fm_obj.localYOLOAnnotationDir + 'GenericSex/')
		model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
		results = model.train(data=fm_obj.localYOLOAnnotationDir + 'GenericSex/data.yaml', epochs=100, imgsz=640, project = fm_obj.localYOLODir, name = 'GenericSexTest', batch = -1, exist_ok=True)
		fm_obj.uploadData(fm_obj.localYOLODir + 'GenericSexTest/')

def trackData():
	# Check for MPS
	device = 'cpu' if torch.backends.mps.is_available() else 'cpu'
	print(f"Using device: {device}")


	fm_obj = FM(analysisID = 'YH_MC_Parentals')
	test_file = fm_obj.localLabeledDLCClipsDir + 'MCYHF1_549_t011_tr1__0009_vid__DLC.mp4'
	fm_obj.downloadData(test_file)
	fm_obj.downloadData(fm_obj.localYOLOModelDir)
	fm_obj.downloadData(fm_obj.localMLPoseDir + 'Benthics/weights/last.pt')

	model_pose = YOLO(fm_obj.localMLPoseDir + 'Benthics/weights/last.pt')
	model_track = YOLO(fm_obj.localYOLOModelFile)

	results = model_pose.track(test_file, show=True, device=device)

train_model('pose')
train_model('detect')