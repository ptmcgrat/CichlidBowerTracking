import pdb, os
import pandas as pd
from helper_modules.file_manager import FileManager as FM

fm_obj = FM(analysisID = 'YH_MC_Parentals')
s_dt = fm_obj.s_dt

#fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
#fm_obj.downloadData(fm_obj.localLabeledClipsFile)

dt = pd.read_csv(fm_obj.localLabeledClipsFile, index_col = 0)
dt[dt.AnalysisID == 'YH_MC_Parentals']
dt['ClipExists'] = True
dt['ProjectID'] = dt.ClipName.str.split('__').str[0]
dt['ClipName'] = dt.ClipName + '.mp4'

for lid,row in dt.iterrows():
	clip_location = fm_obj.localLabeledClipsDir + row.ClipName
	dt.loc[lid,'ClipExists'] = os.path.exists(clip_location)

missing_projects = dt[dt.ClipExists == False].groupby('ProjectID').count().index.tolist()
pdb.set_trace()