
import subprocess, os

os.environ['HOME'] = 'C:/Users/prera/OneDrive/Desktop/McGrath Lab/CichlidBowerTracking/cichlid_bower_tracking'
home_path = os.getenv('HOME') or os.getenv('USERPROFILE')
print(f'The HOME path is: {home_path}')

args = ['python3', 'runAnalysis.py']

args.extend(['Cluster'])
args.extend(['--Workers', '24'])
# args.extend(['--AnalysisType', 'Cluster'])

subprocess.run(args)