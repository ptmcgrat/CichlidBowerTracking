from helper_modules.file_manager import FileManager as FM
import os, pdb
import pandas as pd

fm_obj = FM()
fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
fm_obj.downloadData(fm_obj.localLabeledClipsFile)

dt = pd.read_csv(fm_obj.localLabeledClipsFile)

for index,row in dt.iterrows():
	video_file_path = os.path.join(fm_obj.inputVideosDir,row.ClipName)
	if not os.path.exists(video_file_path):
		pdb.set_trace()