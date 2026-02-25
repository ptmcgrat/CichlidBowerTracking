import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO

fm_obj = FM(analysisID = 'YH_MC_Parentals')
trial_data = fm_obj.localPoseDir + 'MCxYH-tucker-2025-11-03/'
fm_obj.downloadData(trial_data)

yaml_file = trial_data + 'data.yaml'
model = YOLO("yolo26n-pose.yaml")  # build a new model from YAML
results = model.train(data=yaml_file, epochs = 100, imgsz=640)

