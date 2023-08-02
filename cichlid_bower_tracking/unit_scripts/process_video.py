from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
import argparse, subprocess, os, pdb, datetime

def logPrinter(message, indent = True):
	f = open(fileManager.localProjectDir + 'VideoProcessLog.txt', 'a')
	if indent:
		print('    ' + str(datetime.datetime.now()) + ': ' + str(message), file = f)
	else:
		print(str(datetime.datetime.now()) + ': ' + str(message), file = f)

	f.close()

parser = argparse.ArgumentParser()
parser.add_argument('VideoFile', type = str, help = 'Name of h264 file to be processed')
parser.add_argument('Framerate', type = float, help = 'Video framerate')
parser.add_argument('ProjectID', type = str, help = 'Video framerate')
parser.add_argument('AnalysisID', type = str, help = 'Video framerate')


args = parser.parse_args()

fileManager = FM(projectID = args.ProjectID, analysisID = args.AnalysisID)
lf = open(fileManager.localLogfile, 'a', buffering = 1) # line buffered


if '.h264' not in args.VideoFile:
	logPrinter(args.VideoFile + ' not an h264 file', indent = False)
	raise Exception(args.VideoFile + ' not an h264 file')

# Convert h264 to mp4
if os.path.exists(args.VideoFile.replace('.h264', '.mp4')):
	logPrinter(args.VideoFile.replace('.h264', '.mp4') + ' already exits. Deleting')
	subprocess.run(['rm', '-f', args.VideoFile.replace('.h264', '.mp4')])
command = ['ffmpeg', '-r', str(args.Framerate), '-i', args.VideoFile, '-threads', '1', '-c:v', 'copy', '-r', str(args.Framerate), args.VideoFile.replace('.h264', '.mp4')]
logPrinter('Beginning conversion of video: ' + args.VideoFile.split('/')[-1], indent = False)
logPrinter(command)
print('ProcessVideoInfo: Task: Converting ' + args.VideoFile + ' into mp4,,Time: ' + str(datetime.datetime.now()), file = lf)
ffmpeg_output = subprocess.run(command, capture_output = True)

try:
	assert os.path.isfile(args.VideoFile.replace('.h264', '.mp4'))
	assert os.path.getsize(args.VideoFile.replace('.h264','.mp4')) > os.path.getsize(args.VideoFile)
except:
	logPrinter('mp4 file does not exist or not the right size')
	raise Exception

# Sync with cloud (will return error if something goes wrong)
logPrinter('Beginning upload of mp4 file')
print('ProcessVideoInfo: Task: Uploading ' + args.VideoFile.replace('.h264', '.mp4') + ' to cloud,,Time: ' + str(datetime.datetime.now()), file = lf)
for i in [1,2,3]:
	try:
		fileManager.uploadData(args.VideoFile.replace('.h264', '.mp4'))
	except:
		logPrinter('Upload try ' + str(i) + ' failed.')
		if i == 3:
			raise Exception
logPrinter('Upload successful')
print('ProcessVideoInfo: Task: Deleting ' + args.VideoFile.replace('.h264', '.mp4') + ',,Time: ' + str(datetime.datetime.now()), file = lf)

logPrinter('Deleting videos')
subprocess.run(['rm', '-f', args.VideoFile])
subprocess.run(['rm', '-f', args.VideoFile.replace('.h264', '.mp4')])

logPrinter('Video Processing complete')
