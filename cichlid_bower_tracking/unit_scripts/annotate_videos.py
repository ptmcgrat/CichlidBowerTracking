
from cichlid_bower_tracking.data_preparers.project_preparer import ProjectPreparer as PP

import pdb, datetime, os, subprocess, argparse, random, cv2
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='This command runs HMM analysis on a single row of data.')
parser.add_argument('ProjectID', type = str, help = 'ProjectID to analyze')
parser.add_argument('Number', type = int, help = 'Limit annotation to x number of frames.')
parser.add_argument('Initials', type = str, help = 'Initials to save annotations')

args = parser.parse_args()

if len(args.Initials) != 3:
	print('Initials must be three letters long')
	sys.exit()

pp_obj = PP(args.ProjectID)
pp_obj.manuallyLabelVideos(args.Initials.upper(), args.Number)