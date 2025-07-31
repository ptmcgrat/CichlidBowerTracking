import argparse, shutil
from helper_modules.file_manager import FileManager as FM

parser = argparse.ArgumentParser(description='This script is a helper script to make it easier to edit a logfile.\n\nThe script will download the logfile to a specific locaiton, allow you to edit it, and then upload it back to Dropbox') 
parser.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
parser.add_argument('ProjectID', type=str, help='The projectID you want to analyze')
args = parser.parse_args()

analysisID = args.AnalysisID

# Identify projects to run analysis on
fm_obj = FM(analysisID = args.AnalysisID, projectID = args.ProjectID)
print('The logfile is ready to be edited. Its location is: ')
print(fm_obj.localLogfile)
user_input = input('Type "y" once you are finished editing the logfile. Any other letter will not save the changes.')
if user_input == 'y':
	print('Uploading logfile')
	fm_obj.uploadData(fm_obj.localLogfile)
else:
	print('Discarding changes to logfile')
shutil.rmtree(fm_obj.localProjectDir)