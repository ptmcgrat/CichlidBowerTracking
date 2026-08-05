from helper_modules.file_manager import FileManager as FM
import argparse, pdb, datetime, subprocess

parser = argparse.ArgumentParser(description='This script fixes some issues with the pipeline') 
parser.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
args = parser.parse_args()
analysisID = args.AnalysisID

# Identify projects to run analysis on
fm_obj = FM(analysisID)
s_dt = fm_obj.s_dt

projectIDs = s_dt[(s_dt.RunAnalysis == True) & (s_dt.Cluster != 'VideoIndices: ')].index.tolist()

for projectID, row in s_dt.loc[projectIDs].iterrows():
	if projectID not in projectIDs:
		continue
	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
	fm_obj.setProjectID(projectID)
	# Add DepthX, DepthY, InFrame to individual video cluster data
	
	videoIndices = row.videoIDsToRun.split(': ')[1].split(',')
	videoAnnotations = row.videoIDsToAnnotate.split(': ')[1].split(',')
	#p_dt = ma_dt[ma_dt.ProjectID == projectID]
	for videoIndex in videoIndices:
		videoObj = fm_obj.returnVideoObject(int(videoIndex))
		fm_obj.downloadData(videoObj.localFishPoseFile)
		command = ['python3', 'unit_scripts/convert_csv_to_parquet.py', videoObj.localFishPoseFile,
		'--project-id',projectID, '--day-index',videoIndex, 
		'--day-label', videoObj.baseName, '--fps', videoObj.framerate, 
		'--out-root', fm_obj.localPoseDir, '--recording-start', videoObj.startTime, '--overwrite']
		command = [str(x) for x in command]
		print(command)
		subprocess.run(command)
	pdb.set_trace()



