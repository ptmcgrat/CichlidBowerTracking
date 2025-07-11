import argparse, sys, atexit
from cichlid_bower_tracking.helper_modules.cichlid_tracker import CichlidTracker as CT

parser = argparse.ArgumentParser(usage='This command starts a script on a Raspberry Pis to collect depth and RGB data. Allows control through a Google Spreadsheet.')

args = parser.parse_args()
	
ct_obj = CT()
atexit.register(ct_obj.cleanup_method)
ct_obj.run()