#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:57:14 2023

@author: bshi
"""


import subprocess
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

cmd1 = "rclone lsf "
cmd2="rclone copy "
cmd3="rm -r "
path='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/AddFishSex/'
trialpaths='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
trackfile='/MasterAnalysisFiles/AllTrackedFish.csv'
Annotationpath='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
videos='/Videos/'
here=' ./'
classes=['Male', 'Female','Other']
#get names of tracks needed.

femalelist=subprocess.check_output(cmd1+path+classes[1], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
fdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'predict_sex', 'annotate_sex'])
for i in femalelist:
    fdf.loc[len(fdf.index)]=i[:-4].split('__')+[classes[1]]

malelist=subprocess.check_output(cmd1+path+classes[0], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
mdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'predict_sex', 'annotate_sex'])
for i in malelist:
    mdf.loc[len(mdf.index)]=i[:-4].split('__')+[classes[0]]

track_annotations = pd.concat([fdf, mdf], axis=0)
df = track_annotations.drop_duplicates(subset = ["trial", "base_name", 'track_id'])

#df1=track_annotations
#df2=df
#common = df1.merge(df2,on=['trial','track_id'])
#print(common)
#df1[(~df1.trial.isin(common.trial))&(~df1.track_id.isin(common.track_id))]

otherlist=subprocess.check_output(cmd1+path+classes[2], shell=True).decode("utf-8").split('\n')[:-1]
#cleaning subprocess output
odf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'predict_sex'])
for i in otherlist:
    odf.loc[len(odf.index)]=i[:-4].split('__')
#df2 = odf.drop_duplicates(subset = ["trial", "base_name", 'track_id'])
bad_annotations=odf.groupby('trial')['trial'].count()
good_annotations=track_annotations.groupby('trial')['trial'].count()

import pandas as pd
import matplotlib.pyplot as plt

df['idx']=df['trial'].apply(lambda x: '_'.join(x.split('singlenuc')[1].split('_')[:2]))
df['Tank']=df['trial'].apply(lambda x: x.split('singlenuc')[1].split('_')[2])
df['acc']=df['annotate_sex']==df['predict_sex']

grouped = df.groupby(['idx'])['acc'].mean()
grouped1 = df.groupby(['idx'])['predict_sex'].count()
#mcounts = grouped.sex

fig, ax = plt.subplots(figsize=(25,5))
ax.bar(grouped.index, grouped)

ax.set_xlabel('Trial')  
ax.set_ylabel('Total accuracy')
ax.set_title(' accuracy by Trial')

plt.show()


#graphs

import numpy as np
means = df.groupby(['idx', 'annotate_sex'])['acc'].mean()
#means = df.groupby(['idx', 'predict_sex'])['predict_sex'].count()
#df['acc']=df['sex']==df['predicted_sex']

male_counts = means.xs('Male', level=1)
female_counts = means.xs('Female', level=1)
for i in set(female_counts.index)-set(male_counts.index):
    male_counts[i]=-1
for i in set(male_counts.index)-set(female_counts.index):
    female_counts[i]=-1
male_counts=male_counts.sort_index()
female_counts=female_counts.sort_index()
index = np.arange(len(female_counts))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(25,5))

rects1 = ax.bar(male_counts.index, male_counts,bar_width, label='Male')
rects2 = ax.bar(index+bar_width, female_counts,bar_width, label='Female')
ax.axhline(y=0.8, color='r', linestyle='--') 
ax.set_title('M/F Accuracy by Trial')
ax.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import itertools
df1=df[df.Tank=='Tk3']
for i in df1['trial'].unique():
    df2=df1[df1.idx=='28_1']
    y_true=list(df2['annotate_sex'])
    y_pred=list(df2['predict_sex'])

    cm = confusion_matrix(y_true, y_pred)
    
    # Define class labels and tick marks
    classes = ['Female', 'Male']
    tick_marks = np.arange(len(classes))  
    
    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: '+str(i))
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add labels to each cell
    thresh = cm.max() / 2.  
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


fm_obj = FM(analysisID = 'Single_nuc_1')
df = pd.read_csv(fm_obj.localLastHourBehaviorFile)
df = df.drop_duplicates()

df['idx']=df['trial'].apply(lambda x: '_'.join(x.split('singlenuc')[1].split('_')[:2]))
df['Tank']=df['trial'].apply(lambda x: x.split('singlenuc')[1].split('_')[2])
de = pd.read_csv(fm_obj.localEuthData, index_col = False)
df = df.set_index('trial')
de = de.set_index('pid')

# Add new column from df_x to df_y
df['behave_or_control'] = de['behave_or_control']
df = df.reset_index()
bdf=df[df.behave_or_control=='B']
cdf=df[df.behave_or_control=='C']

import matplotlib.pyplot as plt
import numpy as np
df['color']=df.behave_or_control.str.lower()

#x = np.arange(3)  # 3 groups
m1 = df.male_sand_drop
f1=df.male_sand_drop+df.female_sand_drop
m2=df.male_feed_spit
f2=df.male_feed_spit+df.female_feed_spit
m3=df.male_feed_multiple
f3=df.male_feed_multiple+df.female_feed_multiple

f=f1+f2+f3
m=m1+m2+m3


index = np.arange(len(m1))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(60,15))

#fig, ax = plt.subplots(figsize=(25,5))
mcolor=list(df.color)
mcolor=['b' if i=='b' else 'b' for i in mcolor]
f1color= ['orange' if i=='b' else 'orange' for i in mcolor]
m2color= ['r' if i=='b' else 'r' for i in mcolor]
f2color= ['gold' if i=='b' else 'gold' for i in mcolor]
m3color= ['pink' if i=='b' else 'pink' for i in mcolor]
f3color= ['purple' if i=='b' else 'purple' for i in mcolor]
rects1 = ax.bar(index, m,bar_width, color=mcolor, label='Male_Feeding')
rects2 = ax.bar(index, f,bar_width, bottom=m, color=f1color, label='Female_Feeding')
#rects3 = ax.bar(index+bar_width, m2,bar_width, color=m2color, label='Male_Feed_Spit')
#rects4 = ax.bar(index+bar_width, f2,bar_width, bottom=m2, color=f2color, label='Female_Feed_Spit')
#rects5 = ax.bar(index+(2*bar_width), m3,bar_width, color=m3color, label='Male_Feed_Multiple')
#rects6 = ax.bar(index+(2*bar_width), f3,bar_width, bottom=m3, color=f3color, label='Female_Feed_Multiple')

#ax.axhline(y=0.8, color='r', linestyle='--') 
ax.set_title("Male and Female Feeding Behaviors")
ax.legend()
ax.set_xticks(index)
ax.set_xticklabels(df.idx)
plt.xticks(fontsize=25)  
plt.yticks(fontsize=30)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
df['color']=df.behave_or_control.str.lower()

#x = np.arange(3)  # 3 groups
m1 = df.male_bower_multi
f1=df.male_bower_multi+df.female_bower_multi
m2=df.male_bower_scoop
f2=df.male_bower_scoop+df.female_bower_scoop
m3=df.male_bower_spit
f3=df.male_bower_spit+df.female_bower_spit

f=f1+f2+f3
m=m1+m2+m3


index = np.arange(len(m1))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(55,15))

#fig, ax = plt.subplots(figsize=(25,5))
mcolor=list(df.color)
mcolor=['b' if i=='b' else 'b' for i in mcolor]
f1color= ['orange' if i=='b' else 'orange' for i in mcolor]
m2color= ['r' if i=='b' else 'r' for i in mcolor]
f2color= ['gold' if i=='b' else 'gold' for i in mcolor]
m3color= ['pink' if i=='b' else 'pink' for i in mcolor]
f3color= ['purple' if i=='b' else 'purple' for i in mcolor]
rects1 = ax.bar(index, m,bar_width, color=mcolor, label='Male_bower')
rects2 = ax.bar(index, f,bar_width, bottom=m, color=f1color, label='Female_bower')
#rects3 = ax.bar(index+bar_width, m2,bar_width, color=m2color, label='Male_bower_scoop')
#rects4 = ax.bar(index+bar_width, f2,bar_width, bottom=m2, color=f2color, label='Female_bower_scoop')
#rects5 = ax.bar(index+(2*bar_width), m3,bar_width, color=m3color, label='Male_bower_spit')
#rects6 = ax.bar(index+(2*bar_width), f3,bar_width, bottom=m3, color=f3color, label='Female_bower_spit')

#ax.axhline(y=0.8, color='r', linestyle='--') 
ax.set_title("Male and Female Bower Behaviors")
ax.legend()
#ax.xticks(fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(df.idx)
plt.xticks(fontsize=25)  
plt.yticks(fontsize=30)
plt.show()