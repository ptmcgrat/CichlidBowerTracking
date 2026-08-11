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

parquet_dir = fm_obj.localUselessDir + 'Parquet/'
cluster_dir = fm_obj.localUselessDir + 'Cluster/'

fm_obj.createDirectory(parquet_dir)
fm_obj.createDirectory(cluster_dir)

for projectID, row in s_dt.loc[projectIDs].iterrows():
	if projectID not in projectIDs:
		continue
	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
	fm_obj.setProjectID(projectID)
	fm_obj.downloadData(fm_obj.localAllLabeledClustersFile)
	# Add DepthX, DepthY, InFrame to individual video cluster data
	
	dt = pd.read_csv(fm_obj.localAllLabeledClustersFile, index_col = 0)
	videoIndices = row.PoseFish.split(': ')[1].split(',')
	for videoIndex in videoIndices:
		videoObj = fm_obj.returnVideoObject(int(videoIndex))
		dt[dt.videoID == videoObj.baseName].to_csv(cluster_dir + projectID + '_' + videoObj.baseName + '.csv')
		fm_obj.downloadData(videoObj.localParquetFile)
		subprocess.run(['mv', videoObj.localParquetFile, parquet_dir + projectID + '_' + videoObj.baseName + '.parquet'])
		
	pdb.set_trace()
