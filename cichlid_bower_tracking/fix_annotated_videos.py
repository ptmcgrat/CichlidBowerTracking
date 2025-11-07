from helper_modules.file_manager import FileManager as FM
import os, pdb, subprocess
import pandas as pd

fm_obj = FM()

fm_obj.getCloudFiles(fm_obj.localLabeledClipsDir)
projects = [x.replace('.tar','') for x in fm_obj.getCloudFiles(fm_obj.localLabeledClipsDir) if len(x) >0 and x[0] != '.']

fm_obj.downloadData(fm_obj.localLabeledClipsFile)
dt = pd.read_csv(fm_obj.localLabeledClipsFile)
dt['VideoExists'] = 'True'
dt['ProjectID'] = dt.ClipName.str.split('__').str[0]

for project in projects:
	#fm_obj.downloadData(fm_obj.localLabeledClipsDir + project, tarred=True)
	clips = os.listdir(fm_obj.localLabeledClipsDir + project)
	for index,row in dt[dt.ProjectID == project].iterrows():
		filename = row.ClipName + '.mp4'
		video_file_path = fm_obj.localLabeledClipsDir + project + '/' + filename
		if filename not in clips:
			best_guess = filename.split('vid')[-1]
			other = [x for x in clips if best_guess in x]
			if len(other) == 1:
				pdb.set_trace()
			elif len(other) == 2:
				for o in other:
					if o[0] == '.':
						out = subprocess.run(['rm','-f',fm_obj.localLabeledClipsDir + project + '/' + o], capture_output = True, encoding = 'utf-8')
						if out.returncode != 0:
							pdb.set_trace()
					else:
						out = subprocess.run(['mv','-f',fm_obj.localLabeledClipsDir + project + '/' + o, video_file_path], capture_output = True, encoding = 'utf-8')
						if out.returncode != 0:
							pdb.set_trace()
				dt.loc[dt.ClipName == row.ClipName,'VideoExists'] = 'Fix'
			else:
				if row.AnalysisID == 'OriginalSetup':
					continue
				fm_obj = FM(analysisID = row.AnalysisID, projectID = row.ProjectID)
				video_index = int(filename.split('__')[1].split('_vid')[0]) - 1
				videoObj = fm_obj.returnVideoObject(video_index)
				if not os.path.exists(videoObj.localManualLabelClipsDir):
					fm_obj.downloadData(videoObj.localManualLabelClipsDir, tarred = True)
				project_clips = os.listdir(videoObj.localManualLabelClipsDir)
				new_name = filename.replace(row.ProjectID + '__','')
				if new_name in project_clips:
					out = subprocess.run(['mv','-f',videoObj.localManualLabelClipsDir + new_name, video_file_path], capture_output = True, encoding = 'utf-8')
					if out.returncode != 0:
						pdb.set_trace()
					continue

				dt.loc[dt.ClipName == row.ClipName,'VideoExists'] = 'False'
print(dt.groupby(['AnalysisID','VideoExists']).count())
print(dt[dt.AnalysisID == 'YH_MC_Parentals'].groupby(['ProjectID','VideoExists']).count())
pdb.set_trace()

