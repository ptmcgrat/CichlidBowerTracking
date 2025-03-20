from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
import argparse, GPUtil, os, sys, subprocess, yaml, pdb

# This code ensures that modules can be found in their relative directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisIDs', nargs='+', type = str, help = 'AnalysisIDs to include in the trained model')
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM(args.AnalysisID)

