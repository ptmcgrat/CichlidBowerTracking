import os, subprocess
os.environ['HOME'] = '/Users/pkolipaka3/Desktop/CichlidBowerTracking/cichlid_bower_tracking'
home_path = os.getenv('HOME') or os.getenv('USERPROFILE')
print(f'The HOME path is: {home_path}')

args = ['python3', 'runAnalysis.py']

args.extend(['Cluster'])
args.extend(['--Workers', '24'])

subprocess.run(args)