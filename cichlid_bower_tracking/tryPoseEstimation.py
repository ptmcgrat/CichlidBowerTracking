import subprocess,pdb,shutil,os,csv, datetime
from helper_modules.file_manager import FileManager as FM
from ultralytics import YOLO

projectID = 'MC_874_t011_tr1'
fm_obj = FM(analysisID = 'YH_MC_Parentals', projectID = projectID)
videoObj = fm_obj.returnVideoObject(6)
fm_obj.downloadData(videoObj.localVideoFile)

model = YOLO(fm_obj.localMLPoseDir + 'Benthics_l/weights/best.pt')
results = model.predict(videoObj.localVideoFile, stream=True, conf = 0.005)

for i,result in enumerate(results):
	if result.probs is not None:
		pdb.set_trace()
	next(results)

pdb.set_trace()

directories = ['CVxMC-tucker-2025-11-13/','MC-tucker-2025-11-03/','MCxYH-tucker-2025-11-03/','PD-tucker-2025-11-10/','YH-tucker-2025-11-03/']
directories = ['Benthics/']
fm_obj = FM(analysisID = 'YH_MC_Parentals')
fm_obj.uploadData(fm_obj.localMLPoseDir + 'Benthics/')
pdb.set_trace()
for directory in directories:
	trial_data = fm_obj.localPoseDir + directory
	fm_obj.downloadData(trial_data)

	yaml_file = trial_data + 'data.yaml'
	model = YOLO("yolo26l-pose.pt")  # build a new model from YAML
	results = model.train(data=yaml_file, epochs = 200, imgsz=640, project = fm_obj.localMLPoseDir, name=directory.replace('/','_l/'), batch=-1)
	fm_obj.uploadData(fm_obj.localMLPoseDir + directory.replace('/','_l/'))
	fm_obj.uploadData(fm_obj.localMLPoseDir + directory)