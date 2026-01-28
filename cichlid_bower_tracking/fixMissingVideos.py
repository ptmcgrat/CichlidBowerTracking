import pdb
import pandas as pd
from helper_modules.file_manager import FileManager as FM

fm_obj = FM(analysisID = 'MC_YH_Parentals')
s_dt = fm_obj.s_dt

fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
fm_obj.downloadData(fm_obj.localLabeledClipsFile)

a_dt = pd.read_csv(fm_obj.localLabeledClipsFile, index_col = 0)
a_dt['ClipExists'] = True

for lid,row in a_dt.iterrows():
	pdb.set_trace()
