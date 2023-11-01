#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:06:49 2022

@author: bshi42
"""
import cv2
import pandas as pd
import itertools
from matplotlib.pyplot import cm
import more_itertools as mit
import pdb
import os

class cut_video:
    '''
    The purpose of this function is to allow the user to select cuts of the video they would like to produce detections on
    
    currently have the follo2wing cut functions: 
    r_behavior
    this give you that one video for each of the behavior types  (each video 5 secounds long)
    behavior is chosen based to be the behavior with the highest p_value
    
    r_time
    allows user to choose a section of the video to generate based on the number of minutes in time
    
    '''
    fps=29

    time=15
    
    
    #def __init__(self):
    
    def r_behavior(self, df):
        df=df.groupby('behavior_class', as_index=False).apply(lambda x: x.iloc[x.behavior_p_value.argmax(),])
        df = df.sort_values(by='sframe')
        fstart=list(df['sframe'])
        fend=list(df['eframe'])
        return df,  fstart, fend
    
    def r_minutes(self, clusterdf):
        #determine possible cuts for a given amount of time
        frames=self.framerate*60*self.time
        maxvalue=clusterdf['sframe'].max()
        sect, rdr=divmod(maxvalue, frames)
        print(' requesting ' +self.time+' minutes of video' )
        print(str(sect)+' possible cuts')
        cut=int(input('choose a cut 1-'+str(sect)))
        startframe=(cut-1)*frames
        endframe=cut*frames
        nclustdf=clusterdf[clusterdf['sframe']>=startframe]
        nclustdf=nclustdf[nclustdf['sframe']<=endframe]
        return nclustdf, startframe, endframe 
    def r_nothing(self, stime, etime ):
        fstart=60*29*stime
        fend=60*29*etime
        return fstart, fend

class create_video:
    
    '''
    This 
    '''
    
    w = 1296
    h = 972
    time=0
    fps=29
    #might be 30
    
    # Instance attribute
    
    def __init__(self, videopath, df_dict, fstart, fend, view=True):
        self.df_dict=df_dict
        self.video=videopath
        self.view=view
        self.framestart=fstart
        self.frameend=fend
        if 'y' in df_dict.keys():
            self.y=True
        else:
            self.y=False
        if's' in df_dict.keys():
            self.s=True
        else:
            self.s=False
        if'a' in df_dict.keys():
            self.a=True
        else:
            self.a=False
        if'c' in df_dict.keys():
            self.c=True
        else:
            self.c=False
        if'p' in df_dict.keys():
            self.p=True
        else:
            self.p=False
        #determine whether you are requesting one or several videos
        if type(self.framestart)==list:
            self.nvideo=True
        else:
            self.nvideo=False
        
    def frame_num(self):
        cap = cv2.VideoCapture(self.video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length
    
    def run_one_visual( self, title):    
        if self.nvideo==True:
            try:
                vidlist=self.video.split('/')
                os.mkdir(vidlist[0]+'/'+vidlist[1]+'/'+title)
                print(f"Folder '{title}' created successfully.")
            except FileExistsError:
                print(f"Folder '{title}' already exists. will add videos to folder")
            for i in range(len(self.framestart)):
                framestart=self.framestart[i]
                frameend=self.frameend[i]
                ntitle=title+str(i)
                name = vidlist[0]+'/'+vidlist[1]+'/'+title+'/'+ntitle+vidlist[2]

                self.write_bbox_vid( name, framestart=framestart, frameend=frameend)

        else:
             vidlist=self.video.split('/')
             name = '/'.join(vidlist[:-1])+title+vidlist[-1]
             print(name)
             self.write_bbox_vid(name, framestart=self.framestart, frameend=self.frameend)
    
    def tracking(self, newdf, frame, count):
        if len(newdf['frame']==str(count))>0:
            sdf=newdf[newdf['frame']==count]
            for index, row in sdf.iterrows():
                colour = row['color']
                thickness =int(row['thickness']) 
                start = (int(row['x1']), int(row['y1']))
                end = (int(row['x2']), int(row['y2']))
                cv2.rectangle(frame, start, end, colour, thickness)
        return frame
    
    def clusters(self, clusterdf, frame, count):
            sdf=clusterdf[clusterdf['sframe']<=count]
            sdf=sdf[sdf['eframe']>=count]
            for index, row in sdf.iterrows(): 
                colour = row['color']
                thickness =int(row['thickness']) 
                start = (int(row['x1']), int(row['y1']))
                end = (int(row['x2']), int(row['y2']))
                cv2.rectangle(frame, start, end, colour, thickness)
                #cv2.putText(frame,row["ClipName"], start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 5)
            return frame
    def lines(self, linedf, sdf, frame, count):
        ldf=linedf[linedf['sframe']<=count]
        ldf=ldf[ldf['eframe']>=count]
        if len(sdf['frame']==str(count))>0:
            sdf=sdf[sdf['frame']==count]
            sdf = sdf[sdf.track_id.isin(list(ldf.track_id))]
        for index, row in ldf.iterrows():
            row1=sdf[sdf.track_id==row['track_id']]
            if not row1.empty:
                colour = row['color']
                thickness =int(row['thickness']) 
                start = (int(row1['x1']), int(row1['y1']))
                end = (int(row['xc']), int(row['yc']))
                cv2.line(frame, start, end, colour, thickness)
        return frame
        
    def write_bbox_vid(self, name, framestart, frameend):
        cap = cv2.VideoCapture(self.video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap.set(cv2.CAP_PROP_POS_FRAMES, framestart)
        out = cv2.VideoWriter(name, fourcc, self.fps, (self.w,self.h))
        
        #newdf=self.sortdf[self.sortdf['frame']>=framestart]
        count=framestart
        while(cap.isOpened()):
         
          ret, frame = cap.read()
          if ret == True:
            if count>=framestart:
                if self.y==True:
                    frame=self.tracking(self.df_dict['y'], frame, count)
                if self.c==True:
                    frame=self.clusters(self.df_dict['c'],frame, count)
                if self.p==True:
                    frame=self.tracking(self.df_dict['p'], frame, count)
                    if self.a==True:
                        frame=self.lines(self.df_dict['a'],self.df_dict['p'], frame, count)
                if self.s==True:
                    frame=self.tracking(self.df_dict['s'], frame, count)
                    if self.p==False:
                        if self.a==True:
                            frame=self.lines(self.df_dict['a'],self.df_dict['s'], frame, count)
                if self.view==True:
                    cv2.imshow('Frame',frame)
                out.write(frame)
                count+=1
            else:
                count+=1
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break
            elif count>=frameend: 
              break
          #coversion for the image is hard coded in the sort 
          # Break the loop
          
          else:
            break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

 


