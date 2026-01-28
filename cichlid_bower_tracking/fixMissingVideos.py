import subprocess,pdb,shutil
from helper_modules.file_manager import FileManager as FM

fm_obj = FM()
s_dt = fm_obj.s_dt

fm_obj.downloadData(fm_obj.localLabeledClipsDir, tarred_subdirs = True)
fm_obj.downloadData(fm_obj.localLabeledClipsFile)

a_dt = pd.read_csv(fm_obj.localLabeledClipsFile, index_col = 0)
a_dt['ClipExists'] = True

for lid,row in a_dt.iterrows():
	pdb.set_trace()

projectIDs = s_dt[(s_dt.RunAnalysis == True)].index.to_list()
print(projectIDs)
for projectID in projectIDs:
	print('Running project: ' + projectID)
	try:
		fm_obj.setProjectID(projectID, print_issues = True)
	except Exception as e:
		print('Skipping this project due to issues with tankresetstartstop')
		continue
	fm_obj.downloadData(fm_obj.localFrameDir, tarred = True)
	for movie in fm_obj.lp.movies:
		fm_obj.downloadData(fm_obj.localProjectDir + movie.pic_file)
	prepDirectory = fm_obj.localPrepDir
	fm_obj.createDirectory(prepDirectory)
	for trial_num,trial in enumerate(fm_obj.lp.trials):
		subprocess.call(['cp', fm_obj.localProjectDir + trial.daylight_frames[0].pic_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'FirstDepth.jpg'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.daylight_frames[0].npy_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'FirstDepth.npy'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.daylight_frames[-1].pic_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'LastDepth.jpg'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.daylight_frames[-1].npy_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'LastDepth.npy'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.movies[0].pic_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'FirstPi.jpg'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.movies[-1].pic_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'LastPi.jpg'])
		subprocess.call(['cp', fm_obj.localProjectDir + trial.reset_frame.npy_file, prepDirectory + 'Trial_' + str(trial_num+1) + 'ResetDepth.npy'])

	fm_obj.uploadData(fm_obj.localPrepDir)
	#shutil.rmtree(fm_obj.localProjectDir)