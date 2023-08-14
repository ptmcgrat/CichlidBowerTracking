import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

# This code ensures that modules can be found in their relative directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type=str, choices=['Prep', 'Depth', 'Cluster', 'ClusterClassification', 'TrackFish', 'AssociateClustersWithTracks', 'Summary'], help='Type of analysis to run')
parser.add_argument('AnalysisID', type = str, help = 'ID of analysis state name')
parser.add_argument('--ProjectIDs', type=str, nargs='+', help='Optional name of projectIDs to restrict the analysis to')
parser.add_argument('--Workers', type=int, help='Number of workers')
args = parser.parse_args()


# Identify projects to run analysis on
fm_obj = FM(args.AnalysisID)
fm_obj.downloadData(fm_obj.localSummaryFile)
if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
    print('Cant find ' + fm_obj.localSummaryFile)
    sys.exit()

summary_file = fm_obj.localSummaryFile # Shorthand to make it easier to read
projectIDs = fm_obj.identifyProjectsToRun(args.AnalysisType, args.ProjectIDs)

if len(projectIDs) == 0:
    print('No projects to analyze')
    sys.exit()

print('This script will analyze the folllowing projectIDs: ' + ','.join(projectIDs))

# Set workers
if args.Workers is None:
    workers = os.cpu_count()
else:
    workers = args.Workers

# To run analysis efficiently, we download and upload data in the background while the main script runs
uploadProcesses = [] # Keep track of all of the processes still uploading so we don't quit before they finish

print('Downloading: ' + projectIDs[0] + ' ' + str(datetime.datetime.now()), flush = True)
subprocess.run(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data', args.AnalysisType, args.AnalysisID, projectIDs[0]])
while len(projectIDs) != 0:
    projectID = projectIDs.pop(0)

    print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
    p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.run_analysis', args.AnalysisType, args.AnalysisID, projectID])

    # Download data for the next project in the background
    if len(projectIDs) != 0:
        print('Downloading: ' + projectIDs[0] + ' ' + str(datetime.datetime.now()), flush = True)
        p2 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data', args.AnalysisType, args.AnalysisID, projectIDs[0]])

    # Pause script until current analysis is complete and data for next project is downloaded
    p1.communicate()
    if p1.returncode != 0:
        print('Error with analysis: Quitting')
        sys.exit()
    try:
        p2.communicate() # Need to catch an exception if only one project is analyzed
        if p2.returncode != 0:
            print('Error with downloading: Quitting')
            sys.exit()
    except NameError:
        pass

    fm_obj.updateSummaryFile(projectID, args.AnalysisType)

    #Upload data and keep track of it
    print('Uploading: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
    #uploadProcesses.append(subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, args.AnalysisID, projectID]))
    uploadProcesses.append(subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, '--Delete', args.AnalysisID, projectID]))

for i,p in enumerate(uploadProcesses):
    print('Finishing uploading process ' + str(i) + ': ' + str(datetime.datetime.now()), flush = True)
    p.communicate()

"""
if args.AnalysisType == 'Summary':
    import PyPDF2 as pypdf
    paths = [x for x in os.listdir(fm_obj.localAnalysisStatesDir) if '_DepthSummary.pdf' in x]
    writer = pypdf.PdfFileWriter()
    for path in paths:
        f = open(fm_obj.localAnalysisStatesDir + path, 'rb')
        reader = pypdf.PdfFileReader(f)
        for page_number in range(reader.numPages):
            writer.addPage(reader.getPage(page_number))
    with open(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf', 'wb') as f:
        writer.write(f)
    print('Finished analysis: ' + str(datetime.datetime.now()), flush = True)
    fm_obj.uploadData(fm_obj.localAnalysisStatesDir + 'Collated_DepthSummary.pdf')
"""
