# To do
# 1. Make summary file on Dropbox
# 2. Handle Sigint to make sure uploads complete

import argparse, subprocess, pdb, datetime, os
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('DataType', type = str, choices=['Videos','Frames'], help = 'Type of analysis to run')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--ProjectIDs', type = str, nargs = '+', help = 'Name of projectIDs to run analysis on')
group.add_argument('--SummaryFile', type = str, help = 'Name of csv file that specifies projects to analyze')
parser.add_argument('-n', '--Number', type = int, help = 'Limit annotation to x number of videos/frames per project.')
parser.add_argument('-p', '--Practice', action = 'store_true', help = 'Use if you dont want to save annotations')
parser.add_argument('-i', '--Initials', required = True, type = str, help = 'Initials to save annotations')

args = parser.parse_args()

numbers = {}
# Identify projects to run analysis on
if args.ProjectIDs is not None:
	projectIDs = args.ProjectIDs # Specified at the command line
	for projectID in projectIDs:
		numbers[projectID] = args.Number
else:
	fm_obj = FM() 
	summary_file = fm_obj.localAnalysisStatesDir + args.SummaryFile
	fm_obj.downloadData(summary_file)
	dt = pd.read_csv(summary_file, index_col = False)

	projectIDs = list(dt.projectID) # Only run analysis on projects that need it
	for projectID in projectIDs:
		if dt.loc[dt.projectID == projectID]['Labeled' + args.DataType] > 0:
			numbers[projectID] = dt.loc[dt.projectID == projectID]['Labeled' + args.DataType]

# To run analysis efficiently, we download and upload data in the background while the main script runs

for projectID, number in numbers.items():
	print('Downloading: ' + projectID + ' ' + str(datetime.datetime.now()))
	subprocess.run(['python', '-m', 'cichlid_bower_tracking.unit_scripts.download_data', 'ManualLabel' + args.DataType, '--ProjectID', projectID])

	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()))

	# Run appropriate analysis script
	if args.DataType == 'Videos':
		subprocess.run(['python', '-m', 'cichlid_bower_tracking.unit_scripts.annotate_videos', projectID, str(number), args.Initials])
	elif args.DataType == 'Frames':
		subprocess.run(['python', '-m', 'cichlid_bower_tracking.unit_scripts.annotate_frames', projectID, str(number), args.Initials])

	#Upload data and keep track of it
	if not args.Practice:
		print('Uploading: ' + projectID + ' ' + str(datetime.datetime.now()))
		subprocess.run(['python', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', 'ManualLabel' + args.DataType, '--Delete', '--ProjectID', projectID])
		#uploadProcesses.append(subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, projectID]))
	else:
		subprocess.run(['python', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', 'ManualLabel' + args.DataType, '--Delete', '--NoUpload', '--ProjectID', projectID])

print('Finished analysis: ' + str(datetime.datetime.now()))
