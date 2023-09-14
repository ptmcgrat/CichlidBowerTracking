#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:44:00 2023

@author: bshi
"""

import subprocess, os, pdb, datetime
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import datetime

class ClusterSexAssociationPreparer():

#rewrite everything after this line 
    def __init__(self, fileManager):

        self.__version__ = '1.0.0'
        self.fm = fileManager

    def validateInputData(self):
        
        assert os.path.exists(self.fm.localLogfileDir)
        assert os.path.exists(self.fm.localAllFishDetectionsFile)
        assert os.path.exists(self.fm.localAllFishTracksFile)
        assert os.path.exists(self.fm.localOldVideoCropFile)
        assert os.path.exists(self.fm.localAllLabeledClustersFile)
    
    
        
    def find_identity(self,clusterdf, sortdf):
        
        #self.clusterdf['frame']=self.clusterdf['sframe']
        #clusterdf.rename(columns={'xc': 'xc1', 'yc': 'yc1'},inplace=True)
        #startdf= pd.merge(self.clusterdf, self.sortdf, on=["frame"])
        #startdf['distance']=np.sqrt( np.array(((startdf['xc']-startdf['xc1'])**2+((startdf['yc']-startdf['yc1']))**2), dtype=np.float64))
        df=pd.DataFrame(columns=list(clusterdf.columns)+['track_id', 'class'])
        nodf=pd.DataFrame(columns=clusterdf.columns)
        for i in clusterdf['base_name'].unique():
            cdf=clusterdf[clusterdf['base_name']==i].copy()
            tdf=sortdf[sortdf['base_name']==i].copy()
            for index, row in cdf.iterrows():
                 startdf=tdf[(tdf['frame']>=row['sframe']) & (tdf.frame<=row['eframe'])].copy()
                 if not startdf.empty:
                     startdf['distance'] = (startdf.xc + startdf.yc * 1j - (row.xc + row.yc * 1j)).abs()
                     startdf['meandist']=startdf.groupby('track_id').distance.transform(lambda x : x.mean())
                     
                     startdf=startdf[startdf.meandist==startdf.meandist.min()]
                     row['track_id'] = list(startdf.track_id.unique())[0]
                     row['class']=list(startdf.class_id.unique())[0]
                     df=pd.concat([df, row.to_frame().T])
                     #MC_singlenuc21_3_Tk53_021220
                     #check this after run
                     
                 else:
                     nodf=pd.concat([nodf, row.to_frame().T])
        #change this
        df.to_csv(self.fm.localAllTracksAsscociationFile)
        return df
    
    def create_summary(self, poly, t_dt, d_dt):
        t_output = [poly.contains(Point(x, y)) for x,y in zip(t_dt.xc, t_dt.yc)]
        t_dt['InBounds'] = t_output
        track_lengths = t_dt.groupby('track_id').count()['base_name'].rename('track_length')
        t_dt = pd.merge(t_dt, track_lengths, left_on = 'track_id', right_on = track_lengths.index)
        t_dt['binned_track_length'] = t_dt.track_length.apply(bin_tracklength)

        try:
            temp = t_dt[t_dt.p_value > .7].groupby(['track_id', 'track_length', 'base_name']).mean()[['class', 'p_value','InBounds']].rename({'class':'SexCall'}, axis = 1).reset_index().sort_values(['base_name','track_id'])
            temp.to_csv(self.fm.localAllTracksSummaryFile, index = False)
            return temp
        except:
            pdb.set_trace()
            
    def runAssociationAnalysis(self):
        video_crop = np.load(self.fm.localOldVideoCropFile)
        poly = Polygon(video_crop)
        
        t_obj=base(self.fm.localAllFishTracksFile)
        t_dt = t_obj.clean_sort()
        c_obj = base(self.fm.localAllLabeledClustersFile)
        c_dt=c_obj.clean_cluster()
        a_dt=self.find_identity(c_dt, t_dt)