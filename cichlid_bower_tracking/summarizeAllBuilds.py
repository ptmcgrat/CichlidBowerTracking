import argparse, pandas, pdb
from helper_modules.file_manager import FileManager as FM
from helper_modules.googleController import GoogleController as GC
import PyPDF2 as pypdf

parser = argparse.ArgumentParser(description='This script is a helper script to make it easier to edit a logfile.\n\nThe script will download the logfile to a specific locaiton, allow you to edit it, and then upload it back to Dropbox') 
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM()
fm_obj.downloadData(fm_obj.localCredentialSpreadsheet)
googleController = GC(fm_obj.localCredentialSpreadsheet, nonPiFlag = True)


writer = pypdf.PdfWriter()

dt = googleController.dt
for index, row in dt.iterrows():
	if row.Status == 'Running':
		fm_obj = FM(analysisID = row.AnalysisID, projectID = row.ProjectID)
		fm_obj.downloadData(fm_obj.localProjectDir + 'CurrentBuild.pdf')
		f = open(fm_obj.localProjectDir + 'CurrentBuild.pdf', 'rb')
		reader = pypdf.PdfReader(f)
		for page_number in range(len(reader.pages)):
			writer.add_page(reader.pages[page_number])

fm_obj.createDirectory(fm_obj.localTankDir)	
with open(fm_obj.localTankDir + 'Collated_CurrentBuild.pdf', 'wb') as f:
	writer.write(f)

fm_obj.uploadData(fm_obj.localTankDir + 'Collated_CurrentBuild.pdf')
