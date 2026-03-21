import argparse, shutil, pdb, subprocess,cv2
from helper_modules.file_manager import FileManager as FM
import pandas as pd

parser = argparse.ArgumentParser(description='This script is a helper script to make it easier to edit a logfile.\n\nThe script will download the logfile to a specific locaiton, allow you to edit it, and then upload it back to Dropbox') 
parser.add_argument('AnalysisID', type=str, help='The AnalysisID you want to analyze')
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM(analysisID = args.AnalysisID)
dt = pd.read_csv("https://docs.google.com/spreadsheets/d/1YdN1R2na-J8AGbFYluA4yPZp2DQOaQ2qQ7vWNfrUstk/export?gid=0&format=csv")
dt = dt.set_index('projectID')
#fm_obj.createDirectory(fm_obj.localAnalysisOutPicsDir)
for projectID in fm_obj.s_dt.index:
	fm_obj.setProjectID(projectID)
	build_photos = fm_obj.getCloudFiles(fm_obj.localBuildPhotosDir)
	for i, trial in enumerate(fm_obj.lp.trials):
		out_photo = fm_obj.lp.sampleID + '_TR_Trial' + str(i+1) + '.jpg'
		if out_photo in build_photos:
			continue
		if trial.tempName in build_photos:
			trial_photo = fm_obj.localBuildPhotosDir + trial.tempName
		elif trial.tempName.replace('jpeg','jpg') in build_photos:
			trial_photo = fm_obj.localBuildPhotosDir + trial.tempName.replace('jpeg','jpg')
		elif trial.tempName.replace('jpeg','.jpeg') in build_photos:
			trial_photo = fm_obj.localBuildPhotosDir + trial.tempName.replace('jpeg','.jpeg')
		elif trial.tempName.replace('jpeg','png') in build_photos:
			trial_photo = fm_obj.localBuildPhotosDir + trial.tempName.replace('jpeg','png')
		else:
			print('Cant find ' + trial.tempName + ' ' + projectID + '_' + fm_obj.lp.sampleID + ' ' + str(i))
			continue
		print(trial_photo)
		continue

		fm_obj.downloadData(trial_photo)
		image = cv2.imread(trial_photo)
		original_height, original_width = image.shape[:2]
		ratio = 800 / original_height
		new_width = int(original_width * ratio)
		resized_image = cv2.resize(image, (new_width, 800), interpolation=cv2.INTER_AREA)
		r_resized = cv2.selectROI("Image", resized_image, showCrosshair=True, fromCenter=False)
		cv2.destroyWindow("Image") # Close the selection window

		# roi is a tuple (x, y, w, h)
		if not any(r_resized):
			print("No ROI selected or selection cancelled.")

		r_original = (
			int(r_resized[0] / ratio),
			int(r_resized[1] / ratio),
			int(r_resized[2] / ratio),
			int(r_resized[3] / ratio)
		)
		imCrop_original = image[int(r_original[1]):int(r_original[1]+r_original[3]), 
					int(r_original[0]):int(r_original[0]+r_original[2])]

		# Display cropped original image (or do whatever you need with it)
		#cv2.namedWindow('Scalable Window', cv2.WINDOW_NORMAL)

		#cv2.resizeWindow('Scalable Window', 800, 600)

		#cv2.imshow('Scalable Window', imCrop_original)
		#cv2.waitKey(0)
		cv2.destroyAllWindows()
		output_path = "cropped_and_resized.jpg"
		cv2.imwrite(fm_obj.localBuildPhotosDir + out_photo, imCrop_original)
		fm_obj.uploadData(fm_obj.localBuildPhotosDir + out_photo)

	