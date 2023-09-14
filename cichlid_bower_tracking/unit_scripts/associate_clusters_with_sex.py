#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:35:45 2023

@author: bshi
"""

import argparse, subprocess
from cichlid_bower_tracking.data_preparers.project_preparer import ProjectPreparer as PP

parser = argparse.ArgumentParser(usage = 'This script will link fish sex to predicted behaviors')
parser.add_argument('ProjectID', type = str, help = 'Manually identify the project you want to analyze')
parser.add_argument('AnalysisID', type = str, help = 'Manually identify the project you want to analyze')

args = parser.parse_args()

pp_obj = PP(projectID = args.ProjectID, analysisID = args.AnalysisID)
pp_obj.runClusterSexAssociationAnalysis()
