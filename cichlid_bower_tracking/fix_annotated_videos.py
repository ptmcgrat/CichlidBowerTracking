from helper_modules.file_manager import FileManager as FM
import os, pdb
import pandas as pd

fm_obj = FM()
fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
fm_obj.downloadData(fm_obj.localLabeledClipsFile)

dt = pd.read_csv(fm_obj.localLabeledClipsFile)
dt['VideoExists'] = True
clips = os.listdir('/home/student/Temp/CichlidAnalyzer/__AnnotatedData/LabeledVideos/Clips/')

for index,row in dt.iterrows():
	filename = row.ClipName + '.mp4'
	video_file_path = os.path.join(fm_obj.localLabeledClipsDir, filename)
	if not os.path.exists(video_file_path):
		best_guess = filename.split('vid')[-1]
		other = [x for x in clips if best_guess in x]
		pdb.set_trace()
