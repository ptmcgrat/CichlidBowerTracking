import argparse, pdb
from cichlid_bower_tracking.data_preparers.project_preparer import ProjectPreparer as PP

parser = argparse.ArgumentParser(usage = 'This script will use a previously trained 3D Resnet model to classify videos')
parser.add_argument('ProjectID', type = str, help = 'Which projectID you want to identify')
parser.add_argument('ModelID', type = str, help = 'Which previously trained ModelID you want to use to classify the videos')
args = parser.parse_args()

pp_obj = PP(args.ProjectID, modelID = args.ModelID)
pp_obj.run3DClassification()

