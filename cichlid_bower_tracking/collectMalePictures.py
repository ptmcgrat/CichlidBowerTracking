import argparse, shutil, pdb, subprocess
from helper_modules.file_manager import FileManager as FM

parser = argparse.ArgumentParser(description='This script is a helper script to make it easier to edit a logfile.\n\nThe script will download the logfile to a specific locaiton, allow you to edit it, and then upload it back to Dropbox') 
parser.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM(analysisID = args.AnalysisID)
fm_obj.createDirectory(fm_obj.localAnalysisOutPicsDir)
for projectID in fm_obj.s_dt.index:
	fm_obj.setProjectID(projectID)
	fm_obj.getCloudFiles(fm_obj.localProjectDir)
	download_files = [x for x in fm_obj.getCloudFiles(fm_obj.localProjectDir) if 'OUT' in x.upper()]
	for download_file in download_files:
		fm_obj.downloadData(fm_obj.localProjectDir + download_file)
		subprocess.run(['mv',fm_obj.localProjectDir + download_file, fm_obj.localAnalysisOutPicsDir])
	
fm_obj.uploadData(fm_obj.localAnalysisOutPicsDir)