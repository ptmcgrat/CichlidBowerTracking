import argparse, pdb, datetime
import pandas as pd

from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
def to_frame(dt_time):   
    return 29*60*60*dt_time.hour + 29*60*dt_time.minute + 29*dt_time.secound



fm_obj = FM(analysisID = 'Single_nuc_1')

fm_obj.downloadData(fm_obj.localEuthData)

df=pd.read_csv(fm_obj.localEuthData)
#print(df.columns)

projectIDs = list(df.pid)
df1 = pd.DataFrame(columns=['trial','video','B/C','video_start', 'dissection_time', 'feed_total', 'feed_male', 'build_total','build_male', 'spawn_total', 'spawn_male' ])

for projectID in projectIDs:
    fm_obj.createProjectData(projectID)
    fm_obj.downloadData(fm_obj.localAllTracksAsscociationFile)
    a_df = pd.read_csv(fm_obj.localAllTracksAsscociationFile)
    logob=fm_obj.lp
    dissection_time=datetime.datetime.strptime(str(df[df.pid==projectID].dissection_time.values[0]), '%m/%d/%Y %H:%M')
    video_start=logob.movies[-1].startTime
    sub_df=a_df[a_df.base_name==logob.movies[-1].baseName]
    #print(dissection_time)
    #print(video_start)
    if video_start.day==dissection_time.day:
        fromend=dissection_time-video_start
        hourfromend=fromend-datetime.timedelta(hours=1)
    else:
        dissection_time=dissection_time.replace(day=video_start.day)
        fromend=dissection_time-video_start
        hourfromend=fromend-datetime.timedelta(hours=1)
    sub_df=sub_df[sub_df.sframe>=int(hourfromend.total_seconds()*29)]
    sub_df=sub_df[sub_df.eframe<=int(fromend.total_seconds()*29)]
    sub1_df=sub_df[sub_df['class']<=0.7]
    feed_df=sub_df[sub_df.Prediction.isin(list('fmt'))]
    feed1_df=sub1_df[sub1_df.Prediction.isin(list('fmt'))]
    spawn_df=sub_df[sub_df.Prediction.isin(list('s'))]
    spawn1_df=sub1_df[sub1_df.Prediction.isin(list('s'))]
    build_df=sub_df[sub_df.Prediction.isin(list('bcp'))]
    build1_df=sub1_df[sub1_df.Prediction.isin(list('bcp'))]
    #print(len(sub_df.index))
    
    df1.loc[len(df1.index)] = [projectID,logob.movies[-1].baseName, df[df.pid==projectID].behave_or_control.values[0], video_start, dissection_time, len(feed_df.index), len(feed1_df.index), len(build_df.index), len(build1_df.index), len(spawn_df.index), len(spawn1_df.index) ]
df1.to_csv('events_done.csv', index = False)
    #pdb.set_trace()
