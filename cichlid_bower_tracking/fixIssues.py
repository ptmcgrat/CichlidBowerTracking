import subprocess,pdb,shutil
from helper_modules.file_manager import FileManager as FM

fm_obj = FM('YHMC_Mapping')
s_dt = fm_obj.s_dt

projectIDs = s_dt[(s_dt.StartingFiles == True)].index.to_list()
for projectID in projectIDs:
	print('Running project: ' + projectID)
	try:
		fm_obj.setProjectID(projectID, print_issues = True)
	except Exception as e:
		print('Skipping this project due to issues with tankresetstartstop')
		continue
	if fm_obj.checkFileExists(fm_obj.localPrepDir + 'Trial_1FirstDepth.jpg'):
		print(projectID + ' already fixed.')
		if fm_obj.checkFileExists(fm_obj.localPrepDir + 'Trial_1ResetDepth.npy/Trial_1ResetDepth.jpg'):
			print('Need to fix names')
			for i,trial in enumerate(fm_obj.lp.trials):
				bad_name = fm_obj.localPrepDir + 'Trial_' +str(i+1)+'ResetDepth.npy/Trial_'+str(i+1)+'ResetDepth.jpg'
				good_name = fm_obj.localPrepDir + 'Trial_' +str(i+1)+'ResetDepth.npy'
				subprocess.call(['rm','-rf',good_name])
				fm_obj.downloadData(bad_name)
				
				subprocess.call(['mv',bad_name, good_name + '1'])
				subprocess.call(['rm', '-rf' , good_name])
				subprocess.call(['mv',good_name + '1', good_name])
				fm_obj.deleteCloudData(good_name)
				fm_obj.uploadData(good_name)
				#print(['rclone','move',bad_name,good_name])
				#subprocess.call(['rclone','move',bad_name,good_name])
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