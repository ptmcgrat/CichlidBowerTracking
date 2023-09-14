import argparse, subprocess, pdb, datetime, os, sys
import pandas as pd
sys.path.append('/data/home/bshi42/CichlidBowerTracking/') 


from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser(
    description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type=str, choices=['Prep', 'Depth', 'Cluster', 'ClusterClassification', 'TrackFish', 'AssociateClustersWithTracks', 'AddFishSex','Summary'],
                    help='Type of analysis to run')
parser.add_argument('AnalysisID', type = str, help = 'ID of analysis state name')
parser.add_argument('--ProjectIDs', type=str, nargs='+', help='Name of projectIDs to restrict analysis to')
parser.add_argument('--Workers', type=int, help='Number of workers')
parser.add_argument('--ModelID', type=str, help='ModelID to use to classify clusters with')
args = parser.parse_args()

device=list(range(7))

def get_projects(fm_obj, analysis_type, fil_projectIDs):
    #fm_obj.downloadData(fm_obj.localSummaryFile)
    dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str,'TrackFish':str, 'AssociateClustersWithTracks':str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})

    # Identify projects to run on:
    sub_dt = dt[dt.RunAnalysis.astype(str).str.upper() == 'TRUE'] # Only analyze projects that are indicated
    if analysis_type == 'Prep':
        sub_dt = sub_dt[sub_dt.StartingFiles.astype(str).str.upper() == 'TRUE'] # Only analyze projects that have the right starting files
    elif args.AnalysisType == 'Depth':
        sub_dt = sub_dt[sub_dt.Prep.astype(str).str.upper() == 'TRUE'] # Only analyze projects that have been prepped
    projectIDs = list(sub_dt[sub_dt[analysis_type].astype(str).str.upper() == 'FALSE'].projectID) # Only run analysis on projects that need it
    # Filter out projects if optional argment given
    if fil_projectIDs is not None:
        for projectID in projectIDs:
            if projectID not in fil_projectIDs:
                projectIDs.remove(projectID)
    return projectIDs

# Identify projects to run analysis on
fm_obj = FM(analysisID = args.AnalysisID)
#fm_obj.downloadData(fm_obj.localSummaryFile)
#fm_obj.downloadData(fm_obj.localEuthData)
if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
    print('Cant find ' + fm_obj.localSummaryFile)
    sys.exit()

summary_file = fm_obj.localSummaryFile # Shorthand to make it easier to read
projectIDs = get_projects(fm_obj, args.AnalysisType, args.ProjectIDs)

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

dt = pd.read_csv(fm_obj.localSummaryFile, index_col = False, dtype = {'StartingFiles':str, 'RunAnalysis':str, 'Prep':str, 'Depth':str, 'Cluster':str, 'ClusterClassification':str, 'TrackFish':str, 'AssociateClustersWithTracks': str, 'LabeledVideos':str,'LabeledFrames': str, 'Summary': str})
de = pd.read_csv(fm_obj.localEuthData, index_col = False)
trialidx={}
for pid in dt.projectID:
    temp_de=de[de.pid==pid]
    pid_et=datetime.datetime.strptime(str(temp_de.dissection_time.values[0]), "%m/%d/%Y %H:%M")
    fm_obj=FM(projectID = pid, analysisID = args.AnalysisID)
    videos = fm_obj.lp.movies
    count=0
    for videoIndex in videos:
        delta=videoIndex.endTime-pid_et
        days=delta.total_seconds() / (60*60*24)
        if days<1:
            trialidx[pid]=count
        count+=1

device=list(range(1))
for i in range(len(projectIDs)):
    if len(projectIDs)<len(device):
        device=list(range(len(projectIDs)))
    for i in device:
        dt.loc[dt.projectID == projectIDs[i],args.AnalysisType] = 'Running'
        dt.to_csv(summary_file, index = False)
        #fm_obj.uploadData(summary_file)
        print('Downloading: ' + projectIDs[i] + ' ' + str(datetime.datetime.now()), flush = True)
        p1=subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data',args.AnalysisType, '--ProjectID', projectIDs[i], '--ModelID', str(args.ModelID), '--AnalysisID', args.AnalysisID, '--VideoIndex', str(trialidx[projectIDs[i]]) ])
    p1.communicate()


    for i in device: 
        projectID = projectIDs[i]
        print('Running: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
        p2 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.add_fish_sex', projectID, args.AnalysisID, '--VideoIndex', str(trialidx[projectIDs[i]]), '--Device', str(i)])
    p2.communicate()

    #projectIDs = get_projects(fm_obj, args.AnalysisType, args.ProjectIDs)
    for i in device: 
        projectID = projectIDs[i]
        print('Uploading: ' + projectID + ' ' + str(datetime.datetime.now()), flush = True)
        uploadProcesses.append(subprocess.Popen(
            ['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, '--Delete',
             '--ProjectID', projectID, '--AnalysisID', args.AnalysisID]))


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
