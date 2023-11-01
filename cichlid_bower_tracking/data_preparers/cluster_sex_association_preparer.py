#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:44:00 2023

@author: bshi
"""

import subprocess, os, pdb, datetime
import pandas as pd
import numpy as np
#from shapely.geometry import Point, Polygon
import datetime



class ClusterSexAssociationPreparer():

#rewrite everything after this line 
    def __init__(self, fileManager):

        self.__version__ = '1.0.0'
        self.fm = fileManager
        self.projectID=self.fm.projectID
        self.clusterdict={'d': 'sand_drop','t': 'feed_spit','m': 'feed_multiple', 'f': 'feed_scoop','s': 'quiver_spawn','b': 'bower_multi','c': 'bower_scoop','p': 'bower_spit', 'o': 'Fishother', 'x': 'NoFishOther'}

    def validateInputData(self):
        
        assert os.path.exists(self.fm.localLastHourFrameFile)
        assert os.path.exists(self.fm.localLastHourBehaviorFile)
        assert os.path.exists(self.fm.localAllTracksAsscociationFile)
        assert os.path.exists(self.fm.localAllFishSexFile)
    
    def col_mean(self, df, name, by):
        means = df.groupby(by)[name].mean().rename('average_'+name)
        merged = df.merge(means, on=by)
        return merged
        
    def find_identity(self,at_df, fs_df):
        cuts=pd.read_csv(self.fm.localLastHourFrameFile)
        cut1=int(cuts[cuts.trial==self.projectID].sframe)
        cut2=int(cuts[cuts.trial==self.projectID].eframe)
        #print(cuts[cuts.trial==self.projectID])
        #print(cut1)
        #print(cut2)
        #print((cut1/29)/60)
        #print((cut2/29)/60)
        
        
        bdf = pd.read_csv(self.fm.localLastHourBehaviorFile)
        
        newdf=pd.DataFrame(columns=list(at_df.columns)+['average_sex_class'])
        for i in fs_df['base_name'].unique():
            #print('base_name')
            #print(i)
            sat_df=at_df[at_df['base_name']==i].copy()
            sfs_df=fs_df[fs_df['base_name']==i].copy()
            sfs_df=self.col_mean(sfs_df, 'sex_class', 'track_id')
            for index, row in sat_df.iterrows():
                sub_sfs_df=sfs_df[sfs_df.track_id==row.track_id]
                row['average_sex_class']=list(sub_sfs_df.average_sex_class.unique())[0]
                newdf=pd.concat([newdf, row.to_frame().T])
        #print(newdf.info)
        newdf=newdf[newdf.eframe>=cut1]
        newdf=newdf[newdf.sframe<=cut2]
        male_df=newdf[newdf.average_sex_class>=0.5]
        female_df=newdf[newdf.average_sex_class<0.5]
        new_row={'trial': self.projectID}
        for i in self.clusterdict.keys():
            if i in set(female_df.Prediction):
                new_row['female_'+self.clusterdict[i]]=female_df.groupby('Prediction')['Prediction'].count()[[i]].sum()
            else:
                new_row['female_'+self.clusterdict[i]]=0
            if i in set(male_df.Prediction):
                new_row['male_'+self.clusterdict[i]]=male_df.groupby('Prediction')['Prediction'].count()[[i]].sum()
            else:
                new_row['male_'+self.clusterdict[i]]=0
        b_list=[]
        for i in self.clusterdict.values():
            val1='male_'+i
            val2='female_'+i
            b_list+=[str(new_row[val1]),str(new_row[val2])]
            
        os.remove(self.fm.localAllMaleBehaviorSummaryFile)
        with open(self.fm.localAllMaleBehaviorSummaryFile, 'a', newline='') as file:
            file.write(','.join([str(new_row['trial'])]+b_list)) 
            file.close()
        return 
    

    def runAssociationAnalysis(self):
        at_df = pd.read_csv(self.fm.localAllTracksAsscociationFile)
        fs_df = pd.read_csv(self.fm.localAllFishSexFile)
        self.find_identity(at_df, fs_df)
        #as_dt.to_csv(self.fm.localAllSexAsscociationFile, index=False)
    