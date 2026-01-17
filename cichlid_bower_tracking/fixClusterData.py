from helper_modules.file_manager import FileManager as FM
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import argparse, pdb, shutil, os, datetime
import pandas as pd

parser = argparse.ArgumentParser(description='This script fixes some issues with the pipeline') 
parser.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
args = parser.parse_args()
analysisID = args.AnalysisID

# Identify projects to run analysis on
fm_obj = FM(analysisID)
s_dt = fm_obj.s_dt

projectIDs = s_dt[(s_dt.RunAnalysis == True) & (s_dt.Cluster != 'VideoIndices: ')].index.tolist()

fm_obj.downloadData(fm_obj.localLabeledClipsFile)
ma_dt = pd.read_csv(fm_obj.localLabeledClipsFile, index_col = 0)
ma_dt['ProjectID'] = ma_dt.ClipName.str.split('__').str[0]

for projectID, row in s_dt.loc[projectIDs].iterrows():
	if projectID not in projectIDs:
		continue
	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
	fm_obj.setProjectID(projectID)
	# Add DepthX, DepthY, InFrame to individual video cluster data
	fm_obj.downloadData(fm_obj.localVideoCropFile)
	fm_obj.downloadData(fm_obj.localTransMFile)
	fm_obj.downloadData(fm_obj.localLabeledClipsProjectDir, tarred = True)
	
	videoIndices = row.videoIDsToRun.split(': ')[1].split(',')
	videoAnnotations = row.videoIDsToAnnotate.split(': ')[1].split(',')
	p_dt = ma_dt[ma_dt.ProjectID == projectID]
	for videoIndex in videoIndices:
		videoObj = fm_obj.returnVideoObject(int(videoIndex))
		#fm_obj.downloadData(videoObj.localAllClipsDir, tarred = True)
		fm_obj.downloadData(videoObj.localLabeledClustersFile)
		fm_obj.downloadData(videoObj.localManualLabelClipsDir, tarred = True)
		
		# Add depthXY
		
		clusterData = pd.read_csv(videoObj.localLabeledClustersFile)
		
		# Add crop information
		with open(fm_obj.localVideoCropFile) as f:
			for line in f:
				video_crop_points = eval(line.rstrip())
		polygon = Polygon(video_crop_points)
		buffered_polygon = polygon.buffer(20, join_style=2)
		clusterData['InFrame'] = clusterData.apply(lambda row: buffered_polygon.contains(Point(row['Y'],row['X'])), axis = 1)
  
		clusterData.to_csv(videoObj.localLabeledClustersFile)
		fm_obj.uploadData(videoObj.localLabeledClustersFile)

		clusterData['NewClipName'] = projectID + '__' + clusterData['ClipName']
		if videoIndex not in videoAnnotations:
			continue
		for i,row in clusterData.iterrows():
			if not row.NewClipName in p_dt.ClipName.tolist():
				continue
			in_video = videoObj.localManualLabelClipsDir + row.ClipName + '_ManualLabel.mp4'
			out_video = fm_obj.localLabeledClipsProjectDir + row.NewClipName + '_ManualLabel.mp4'
			shutil.move(in_video,out_video) #changed for windows
			ma_dt.loc[ma_dt.ClipName == row.NewClipName,'InFrame'] = clusterData.loc[clusterData.NewClipName == row.NewClipName,'InFrame'].values[0]
		
	fm_obj.uploadData(fm_obj.localLabeledClipsProjectDir, tarred = True)	
	shutil.rmtree(fm_obj.localProjectDir)
	shutil.rmtree(fm_obj.localLabeledClipsProjectDir)

ma_dt.to_csv(fm_obj.localLabeledClipsFile)
fm_obj.uploadData(fm_obj.localLabeledClipsFile)

