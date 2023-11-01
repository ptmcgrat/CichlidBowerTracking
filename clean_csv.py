#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:28:41 2022

@author: bshi42
"""

import pandas as pd

import seaborn as sns

import more_itertools as mit

import pdb
import numpy as np


class color_df:
 
    '''
    color can be used to show a property of a detection in a video
    
    input dataframe 
    output dataframe with color column
    
    this function has mutiple methods for adjusting the thickness of a detection in visualization:
    r_unique
    this will loop colors by unique elements of a column
    can be used for track_id in sort
    
    r_class
    takes in a column name and gives color pink to rows with value <=0.5 and blue to values >=0.5
    used for male /female and reflection/fish
    
    r_behavior (for clusterdf)
    gives behavior detections a color based of the type (feeding, building, or spawning)
    
    r_nothing 
    gives all detections a standard black color
    
    '''
    #this may need to consider switch if x y values not equal

    rgb_colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyan
    (128, 0, 0),   # Maroon
    (0, 128, 0),   # Green (dark)
    (0, 0, 128),   # Navy
    (128, 128, 128) # Gray
    ]
    Male =(255,0,0)
    Female=(180, 105,255 )
    Unknown=(128,0,128)
    black=(0,0,0)
    feed=(0,0,0)
    build=(0,0,255)
    spawn=(0,255,0)
    # Instance attribute

    def r_unique(self, df, name):
        num_groups = df[name].nunique()
        color_indices = np.tile(np.arange(len(self.rgb_colors)), int(np.ceil(num_groups / len(self.rgb_colors))))[:num_groups]
        df['color'] = df[name].map(dict(zip(df[name].unique(), np.array(self.rgb_colors)[color_indices])))
        df['color']= df['color'].apply(lambda x: tuple(int(y) for y in x))
        return df
        
        
    def r_class(self, df, name):
        fdf=df[df[name]<=0.5]
        mdf=df[df[name]>0.5]
        mdf['color']=[self.Female]*len(mdf.index)
        fdf['color']=[self.Male]*len(fdf.index)
        df=pd.concat([mdf, fdf], ignore_index=True)
        df=df.sort_values(['frame'])
        return df
        
    def r_behavior(self, clusterdf):
        #separate by behavior type
        feeddf=clusterdf[clusterdf['behavior_class'].isin(list('fmt'))]
        builddf=clusterdf[clusterdf['behavior_class'].isin(list('bcp' ))]
        spawndf=clusterdf[clusterdf['behavior_class'].isin(list('s' ))]
        #color by behavior type
        feeddf['color']=[self.feed]*len(feeddf['behavior_class'])
        builddf['color']=[self.build]*len(builddf['behavior_class'])
        spawndf['color']=[self.spawn]*len(spawndf['behavior_class'])
        clusterdf=pd.concat([feeddf, builddf, spawndf], ignore_index=True)
        return clusterdf
    
    def r_nothing(self, df):
        df['color']=[self.black]*len(df.index)
        return df
    

class thick_df:
 
    '''
    Thickness can be used to show a property of a detection in a video
    
    input dataframe 
    output dataframe with thickness column
    
    this function has mutiple methods for adjusting the thickness of a detection in visualization:
    r_reflection
    this means the thickness will vary with respect to yolo detections (relfection, fish)
    
    r_cbound
    this means the thickness will vary with respect to the calculated InBounds parameter (applicable to sort outputs)
    
    r_nothing 
    gives all detections a standard thickness of 4
    
    r_small
    gives all detections a small thickness of 2
    '''
    #this may need to consider switch if x y values not equal

    inthick=4
    outthick=9
    smallthick=2
    # Instance attribute

    def r_reflection(self, df):
        rdf=df[df['average_reflection_class']>=0.5]
        bdf=df[df['average_reflection_class']<0.5]
        rdf['thickness']=[self.outthick]*len(rdf.index)
        bdf['thickness']=[self.inthick]*len(bdf.index)
        df=pd.concat([rdf, bdf], ignore_index=True)
        df=df.sort_values(['frame', 'track_id'])
        return df
    def r_cbound(self, df):
        rdf=df[df['InBounds']=='False']
        bdf=df[df['InBounds']=='True']
        rdf['thickness']=[self.outthick]*len(rdf.index)
        bdf['thickness']=[self.inthick]*len(bdf.index)
        df=pd.concat([rdf, bdf], ignore_index=True)
        df=df.sort_values(['frame', 'track_id'])
        return df
    def r_nothing(self, df):
        df['thickness']=[self.inthick]*len(df.index)
        return df
    def r_small(self, df):
        df['thickness']=[self.smallthick]*len(df.index)
        return df
    
    


class clean_csv:
 
    '''
    input: 
    dictionary {csv_type: csv_path, ...}
    
    Output: 
    dictionart{csv_type: dataframe}
    
    Current csv_types accepted:
    's': sort detection
    'c': cluster detections
    'y':yolo detectections
    '+': summary file
    'p': pytorch detections
    'l': linedf associates a behavior with 5 secounds of track  this dataframe is generated from either 'c' and ('s' or 'p')
    
    This class does the following (non-visualization format):
    1.) takes in a dictionary of csvs (for visualization)
    2.) converts the csvs to a comparable dataframe format (many csvs have dif name for the same thing)
    3.) returns all dataframes in a dictionary
    
    If vis=True
    all csv will have a thickness and color column added
    see thickness_df and color_df  for more information on their formating
    
    NOTE: this class is highly sensitive to column names, but it is necessary to overcome this in order to do
    comparative detection analysis. If csv column names are edited this script will need to be revised for compatibility
     
    
    '''
    #could change with new videos
    framerate=29
    IMG_W = 1296
    IMG_H = 972
    
    #for cluster boxes (makes 5 sec square box)
    tdelta=framerate*2
    ydelta=60
    xdelta=60

    def __init__(self, csvdict, base_name, vis=True):
        self.dfs={}
        self.bn=base_name
        for i in csvdict.keys():
            df=pd.read_csv(csvdict[i], index_col=0)
            if i =='c':
                print(df[df.LID==198])
                df=self.clean_cluster(df, vis)
            elif i=='s':
                df=pd.read_csv(csvdict[i])
                df=self.clean_sort(df, vis)
            elif i=='y':
                df=self.clean_yolo(df, vis)
            elif i =='+':
                df=pd.read_csv(csvdict[i])
                df=self.clean_sum(df, vis)
            elif i=='p':
                df=self.clean_pytorch(df, vis)
            elif i=='a':
                df=self.clean_association(df, vis)
            else: 
                print('input invalid:  '+i)
            df=df[df['base_name']==self.bn]
            self.dfs[i]=df
    
    def col_mean(self, df, name, by):
        means = df.groupby(by)[name].mean().rename('average_'+name)
        merged = df.merge(means, on=by)
        return merged
    
    def output(self): 
        return self.dfs
    
    def clean_cluster(self, df, vis):
        #standardize naming conventions
        df['base_name']=df['VideoID'].copy()
        df['behavior_class']=df['Prediction'].copy()
        df['behavior_p_value']=df['Confidence'].copy()
        df['behavior_id']=df['LID'].copy()
        
        #filter for when behaviors were predicted
        df=df[df['ClipCreated']=='Yes'] 
        
        #this is added to better understand what X and Y are
        df['yc']=df['X'].copy()
        df['xc']=df['Y'].copy()

        #convert to retangle points
        df['x1']=df['xc']-self.xdelta
        df['y1']=df['yc']-self.ydelta
        df['x2']=df['xc']+self.xdelta
        df['y2']=df['yc']+self.ydelta
        
        #df gives us 't' in secounds
        #we need to convert 't' to frame
        df['frame']=df['t']*self.framerate
        # we construct the behavior as occuring 2 secs before and after this frame
        df['sframe']=(df['frame'] - self.tdelta).astype('int64')
        df['eframe']= (df['frame'] + self.tdelta).astype('int64')
        
        if vis==True: 
            df=thick_df().r_nothing(df)
            df=color_df().r_behavior(df)
        return df   
    def clean_yolo(self, df, vis):
        #standardize naming conventions
        df['reflection_class']=df['class'].copy()
        df['reflection_p_value']=df['p-value'].copy()
        
        #yolo coordinates seemed to be converted incorrectly 
        #this is a weird fix
        y1=(df['x2'].copy())
        x2=(df['y1'].copy())
        df['x2']=x2
        df['y1']=y1
        if vis==True: 
            df=thick_df().r_small(df)
            df=color_df().r_class(df, 'reflection_class')
        return df
    
    def clean_sum(self, df, vis):
        #standardize naming conventions
        df['average_reflection_class']=df['Reflection'].copy()
        df['reflection_p_value']=df['p_value'].copy()
        
        return df

    def clean_sort(self, df, vis):
        #standardize naming conventions
        df['reflection_class']=df['class_id'].copy()
        df['reflection_p_value']=df['p_value'].copy()
        
        #sort conversion to rectangular form
        df['x1']=self.IMG_W*(df['xc']-0.5*df['w'])
        df['y1']=self.IMG_H*(df['yc']-0.5*df['h'])
        df['x2']=self.IMG_W*(df['xc']+0.5*df['w'])
        df['y2']=self.IMG_H*(df['yc']+0.5*df['h'])
        df=self.col_mean(df, 'reflection_class', 'track_id')
        if vis==True: 
            df=thick_df().r_reflection(df)
            #df=color_df().r_class(df, 'average_reflection_class')
            df=color_df().r_unique(df, 'track_id')
        return df
    
    def clean_pytorch(self, df, vis):
        df['reflection_class']=df['class_id'].copy()
        df['reflection_p_value']=df['p_value'].copy()
        
        #sort conversion to rectangular form
        if (df['xc'] <= 1).all() and (df['yc'] <= 1).all():
            df['x1']=self.IMG_W*(df['xc']-0.5*df['w'])
            df['y1']=self.IMG_H*(df['yc']-0.5*df['h'])
            df['x2']=self.IMG_W*(df['xc']+0.5*df['w'])
            df['y2']=self.IMG_H*(df['yc']+0.5*df['h'])
            df['xc']=self.IMG_W*df['xc']
            df['yc']=self.IMG_H*df['yc']
        else:
            df['x1']=(df['xc']-0.5*df['w'])
            df['y1']=(df['yc']-0.5*df['h'])
            df['x2']=(df['xc']+0.5*df['w'])
            df['y2']=(df['yc']+0.5*df['h'])
        #fix this in pytorch 
        df=self.col_mean(df, 'sex_class', 'track_id')
        
        if vis==True: 
            df=thick_df().r_nothing(df)
            #fix this later
            df=color_df().r_class(df, 'average_sex_class')
        return df
    
    def clean_association(self, df, vis):
        df['behavior_class']=df['Prediction'].copy()
        df['behavior_p_value']=df['Confidence'].copy()
        df['behavior_id']=df['LID'].copy()
        df['average_reflection_class']=df['class'].copy()
        
        df=df[df['ClipCreated']=='Yes']
        
        df['yc1']=df['X'].copy()
        df['xc1']=df['Y'].copy()
        if vis==True:
            df=thick_df().r_nothing(df)
            df=color_df().r_nothing(df)
        return df

