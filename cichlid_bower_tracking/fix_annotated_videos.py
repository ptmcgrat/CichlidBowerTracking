from helper_modules.file_manager import FileManager as FM
import os, pdb
import pandas as pd

fm_obj = FM()
fm_obj.downloadData(self.fileManager.localLabeledClipsDir, tarred_subdirs = True)
fm_obj.downloadData(self.fileManager.localLabeledClipsFile)

dt = pd.read_csv(self.fileManager.localLabeledClipsFile)

for index,row in dt.iterrows():
	pdb.set_trace()