import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO
import torch

"""
# Check for MPS
device = 'cpu' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


fm_obj = FM(analysisID = 'YH_MC_Parentals')
test_file = fm_obj.localLabeledDLCClipsDir + 'MCYHF1_549_t011_tr1__0009_vid__DLC.mp4'
fm_obj.downloadData(test_file)
fm_obj.downloadData(fm_obj.localYOLOModelDir)
fm_obj.downloadData(fm_obj.localMLPoseDir + 'Benthics/weights/best.pt')

model_pose = YOLO(fm_obj.localMLPoseDir + 'Benthics/weights/best.pt')
model_track = YOLO(fm_obj.localYOLOModelFile)

results = model_pose.track(test_file, show=True, device=device)
"""
# Train model

fm_obj = FM(analysisID = 'YH_MC_Parentals')

fm_obj.downloadData(fm_obj.localPoseDir + 'Benthics/')
model = YOLO("yolo26x-pose.pt")  # load a pretrained model (recommended for training)

results = model.train(data=fm_obj.localPoseDir + 'Benthics/data.yaml', epochs=100, imgsz=640, project = fm_obj.localMLPoseDir, name = 'Benthics', batch = -1, exist_ok=True)
fm_obj.uploadData(fm_obj.localMLPoseDir + 'Benthics/')
