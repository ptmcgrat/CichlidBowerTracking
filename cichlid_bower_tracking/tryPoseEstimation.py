import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO

directories = ['CVxMC-tucker-2025-11-13/','MC-tucker-2025-11-03/','MCxYH-tucker-2025-11-03/','PD-tucker-2025-11-10/','YH-tucker-2025-11-03/']
directories = ['Benthics/']
fm_obj = FM(analysisID = 'YH_MC_Parentals')

for directory in directories:
	trial_data = fm_obj.localPoseDir + directory
	fm_obj.downloadData(trial_data)

	yaml_file = trial_data + 'data.yaml'
	model = YOLO("yolo26n-pose.pt")  # build a new model from YAML
	results = model.train(data=yaml_file, epochs = 200, imgsz=640, project = fm_obj.localMLPoseDir, name=directory)
	fm_obj.uploadData(fm_obj.localMLPoseDir + directory)