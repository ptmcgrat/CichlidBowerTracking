import os, sys, datetime, subprocess
import pandas as pd
from helper_modules.file_manager import FileManager as FM
import math
import pdb
import shutil, random, fnmatch

analysisId = 'MC_multi'
AnalysisType = 'Cluster'

home_path = os.getenv('HOME') or os.getenv('USERPROFILE')
print(f'The HOME path is: {home_path}')

random.seed(42)

def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text

def is_nan(value):
    if isinstance(value, float):
        return math.isnan(value)
    elif isinstance(value, str):
        return value.lower() == "nan"
    return False

# summaryFile = '/Users/pkolipaka3/Desktop/cichlidData/MC_multi_PM.csv'
fmObj = FM()
# fmObj.downloadData(fmObj.localSummaryFile)

if not fmObj.checkFileExists(fmObj.localSummaryFile):
	print('Cant find ' + fmObj.localSummaryFile)
	sys.exit()
     
fmObj.s_dt.columns = [c.replace(' ', '_') for c in fmObj.s_dt.columns]
fmObj.s_dt.reset_index(inplace=True)

ref_df = pd.read_csv(home_path+"Temp/CichlidAnalyzer/__ProjectData/MC_multi/MC_s15_tr2_BowerBuilding/Troubleshooting/0001_vid_labeledClusters.csv")
selected_video_clips_df = pd.DataFrame({col: pd.Series(dtype=ref_df[col].dtype) for col in ref_df.columns})
selected_video_clips_df["ProjectID"] = ""
print(selected_video_clips_df)

for i, row in fmObj.s_dt.iterrows():
     
    if(is_nan(row.VideoIDs_new)):
        continue

    videoIDs = row.VideoIDs_new.split(': ')[1].split(',')
    print(videoIDs)
    video_folder = home_path +'Temp/CichlidAnalyzer/__ProjectData/MC_multi/'+row.projectID+'/MLClips'
    troubleshooting_folder = home_path +'Temp/CichlidAnalyzer/__ProjectData/MC_multi/'+row.projectID+'/Troubleshooting'
    n_videos = 200//len(videoIDs)
    count = 0
    for video in videoIDs:

        video_number = str(int(video)+1).zfill(4)
        clips_folder = video_folder+'/'+video_number+'_vid/'
        pattern = "*_ManualLabel.mp4"
        clips = [f for f in os.listdir(clips_folder) if not fnmatch.fnmatch(f, pattern)]
        count = 0
        
        selected_video_clips = home_path+'Temp/CichlidAnalyzer/__AnnotatedData/MC_multi/'+row.projectID+'/'+video_number+'/'
        os.makedirs(selected_video_clips, exist_ok=True)
        troubleshooting_df = pd.read_csv(troubleshooting_folder+'/'+video_number+'_vid_labeledClusters.csv', index_col = 0)
        troubleshooting_df['TimeStamp'] = pd.to_datetime(troubleshooting_df['TimeStamp'])

        earliest_time = troubleshooting_df['TimeStamp'].min()
        latest_time = troubleshooting_df['TimeStamp'].max()

        if (latest_time.time() <= pd.to_datetime("9:00:00").time()) or (earliest_time.time()>= pd.to_datetime("19:00:00").time()):
            n_videos = 200//(len(videoIDs) - 1)
            continue

        if clips:
            while count<=n_videos:
                random_clip = random.choice(clips)
                clip_row = troubleshooting_df[troubleshooting_df['ClipName'] == remove_suffix(random_clip,".mp4")]
                startTime = clip_row.TimeStamp
                
                if (startTime.dt.time >= pd.to_datetime("9:00:00").time()).any() and (startTime.dt.time <= pd.to_datetime("19:00:00").time()).any():
                    clip_path = clips_folder+random_clip
                    # if os.path.exists(clip_path):
                    #     print("clip path exists")
                    # else:
                    #     print("clip path doesn't exist")

                    manual_label_clip_path = clips_folder+remove_suffix(random_clip,".mp4")+'_ManualLabel.mp4'
                    # if os.path.exists(manual_label_clip_path):
                    #     print("manual_label_clip_path exists")
                    # else:
                    #     print("manual_label_clip_path doesn't exist")
                    shutil.copy(clip_path, os.path.join(selected_video_clips, os.path.basename(clip_path)))
                    shutil.copy(manual_label_clip_path, os.path.join(selected_video_clips, os.path.basename(manual_label_clip_path)))
                    clip_row = clip_row.copy()
                    clip_row["ProjectID"] = row.projectID
                    selected_video_clips_df = pd.concat([selected_video_clips_df,clip_row], ignore_index=True)
                    # pdb.set_trace()
                    count+=1
        tarfile_path = home_path+'Temp/CichlidAnalyzer/__AnnotatedData/MC_multi/'+row.projectID+'/'+video_number+'.tar'
        cloud_path = 'ptm_dropbox:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/MC_multi/'+row.projectID
        try:
            # Construct the tar command
            command = ["tar", "-czf",tarfile_path, "-C", selected_video_clips, "."]
            
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Folder {selected_video_clips} has been successfully archived as {video_number}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while creating tar archive: {e}")


        output = subprocess.run(['rclone', 'copy', tarfile_path, cloud_path], capture_output = True, encoding = 'utf-8')
        # pdb.set_trace()

cloud_path = 'ptm_dropbox:/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/MC_multi/'
selected_video_clips_df.to_csv(home_path+"Temp/CichlidAnalyzer/__AnalysisStates/MC_multi/selected_video_clips.csv", index=False) 
output = subprocess.run(['rclone', 'copy', home_path+"Temp/CichlidAnalyzer/__AnalysisStates/MC_multi/selected_video_clips.csv", cloud_path], capture_output = True, encoding = 'utf-8')