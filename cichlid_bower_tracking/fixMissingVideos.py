import pdb, os, subprocess
import pandas as pd
from helper_modules.file_manager import FileManager as FM

fm_obj = FM(analysisID = 'YH_MC_Parentals')
s_dt = fm_obj.s_dt

#fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
#fm_obj.downloadData(fm_obj.localLabeledClipsFile)

dt = pd.read_csv(fm_obj.localLabeledClipsFile, index_col = 0)
dt = dt[dt.AnalysisID == 'YH_MC_Parentals']
dt['ClipExists'] = True
dt['ProjectID'] = dt.ClipName.str.split('__').str[0]
dt['ClipName'] = dt.ClipName + '.mp4'
dt['VideoInfo'] = dt.ClipName.str.split('__').str[1]

for lid,row in dt.iterrows():
	clip_location = fm_obj.localLabeledClipsDir + row.ClipName
	dt.loc[lid,'ClipExists'] = os.path.exists(clip_location)
print(len(dt[dt.ClipExists == False]))
missing_projects = dt[dt.ClipExists == False].groupby('ProjectID').count().index.tolist()
for projectID in missing_projects:
	videoIndices = [int(x) for x in s_dt.loc[projectID,'videoIDsToAnnotate'].split(': ')[1].split(',')]
	fm_obj.setProjectID(projectID)
	for videoIndex in videoIndices:
		videoObj = fm_obj.returnVideoObject(videoIndex)
		fm_obj.downloadData(videoObj.localManualLabelClipsDir, tarred = True)
for lid,row in dt[dt.ClipExists == False].iterrows():
	clip_location1 = fm_obj.localLabeledClipsDir + row.ClipName
	clip_location2 = videoObj.localManualLabelClipsDir + row.ClipName.replace(row.ProjectID + '__','')
	if os.path.exists(clip_location2):
		subprocess.run(['cp', clip_location2,clip_location1])

#for projectID in missing_projects:
#	fm_obj.uploadData(fm_obj.localLabeledClipsDir + projectID, tarred = True)