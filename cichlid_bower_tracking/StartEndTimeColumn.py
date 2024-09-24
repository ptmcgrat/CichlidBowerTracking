import os, subprocess, sys, pdb
from helper_modules.file_manager import FileManager as FM
import math
from helper_modules.log_parser import LogParser as LP

os.environ['HOME'] = '/Users/pkolipaka3/Desktop/CichlidBowerTracking/cichlid_bower_tracking'
home_path = os.getenv('HOME') or os.getenv('USERPROFILE')
print(f'The HOME path is: {home_path}')

config_file_path = '/Users/pkolipaka3/.config/rclone/rclone.conf'


analysisId = 'MC_multi'
AnalysisType = 'Cluster'

def is_nan(value):
    if isinstance(value, float):
        return math.isnan(value)
    elif isinstance(value, str):
        return value.lower() == "nan"
    return False

# summaryFile = '/Users/pkolipaka3/Desktop/cichlidData/MC_multi_PM.csv'
fmObj = FM()
fmObj.downloadData(fmObj.localSummaryFile)




if not fmObj.checkFileExists(fmObj.localSummaryFile):
	print('Cant find ' + fmObj.localSummaryFile)
	sys.exit()

fmObj.s_dt.columns = [c.replace(' ', '_') for c in fmObj.s_dt.columns]
fmObj.s_dt['StartTime'] = None
fmObj.s_dt['EndTime'] = None
fmObj.s_dt.reset_index(inplace=True)

# fmObj.s_dt['ProjectID'] = None
pdb.set_trace()

for i, row in fmObj.s_dt.iterrows():
	# print(row.VideoIDs_new)
	# pdb.set_trace()
	# pdb.set_trace()
	if(is_nan(row.VideoIDs_new)):
		continue

	output = subprocess.run(['rclone', 'copy', 'cichlidData:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__ProjectData/MC_multi/'+row.projectID+'/Logfile.txt', home_path+'/Temp/CichlidAnalyzer/__ProjectData/'+row.projectID, '--config', config_file_path],capture_output = True, encoding = 'utf-8')
	# pdb.set_trace()
    
	log_file = home_path+'/Temp/CichlidAnalyzer/__ProjectData/'+row.projectID+'/Logfile.txt'
	logParser = LP(log_file)
	# pdb.set_trace()
	# print(logParser.movies)
	videoIDs = row.VideoIDs_new.split(': ')[1].split(',') 
	startTime = logParser.movies[int(videoIDs[0])].startTime
	endTime = logParser.movies[int(videoIDs[-1])].endTime
    
	
	fmObj.s_dt.at[i,'StartTime'] = startTime
	fmObj.s_dt.at[i,'EndTime'] = endTime	
	# pdb.set_trace()	
	# fmObj.s_dt = fmObj.s_dt[['ProjectID'] + [col for col in fmObj.s_dt.columns if col != 'ProjectID']]

pdb.set_trace()
fmObj.s_dt.to_csv('test.csv', index=False)





