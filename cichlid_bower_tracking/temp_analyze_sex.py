import pdb,subprocess
import pandas as pd
import numpy as np
import seaborn as sns
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from collections import defaultdict

from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

fm_obj = FM('Single_nuc_1')
fm_obj_b = FM('Single_nuc_1',masterDir = 'McGrath/PublicIndividualData/Breanna/Single_nuc_1/') 
fm_obj.downloadData(fm_obj.localEuthData)

pids = set()

females = defaultdict(int)
fem_data = subprocess.run(['rclone', 'lsf', 'CichlidPiData:McGrath/Apps/CichlidPiData/__AnnotatedData/PatrickTesting/MaleFemale/Female'], capture_output = True, encoding = 'utf-8')
for vid in fem_data.stdout.split():
	if '.mp4' not in vid:
		continue
	else:
		pids.add(vid.split('_T')[0])
		females[vid.split('_T')[0]] += 1
males = defaultdict(int)
male_data = subprocess.run(['rclone', 'lsf', 'CichlidPiData:McGrath/Apps/CichlidPiData/__AnnotatedData/PatrickTesting/MaleFemale/Male'], capture_output = True, encoding = 'utf-8')
for vid in male_data.stdout.split():
	if '.mp4' not in vid:
		continue
	else:
		pids.add(vid.split('_T')[0])
		males[vid.split('_T')[0]] += 1

mm = defaultdict(int)
mf = defaultdict(int)
fm = defaultdict(int)
ff = defaultdict(int)

fem_data = subprocess.run(['rclone', 'lsf', 'CichlidPiData:McGrath/Apps/CichlidPiData/__AnnotatedData/AddFishSex/Female'], capture_output = True, encoding = 'utf-8')
for vid in fem_data.stdout.split():
	if '.mp4' not in vid:
		continue
	else:
		pid = vid.split('_T')[0]
		pids.add(vid.split('_T')[0])

		if 'Female' in vid:
			ff[pid] += 1
		else:
			fm[pid] += 1
male_data = subprocess.run(['rclone', 'lsf', 'CichlidPiData:McGrath/Apps/CichlidPiData/__AnnotatedData/AddFishSex/Male'], capture_output = True, encoding = 'utf-8')
for vid in male_data.stdout.split():
	if '.mp4' not in vid:
		continue
	else:
		pid = vid.split('_T')[0]
		pids.add(vid.split('_T')[0])

		if 'Female' in vid:
			mf[pid] += 1
		else:
			mm[pid] += 1

print('ProjectID\tAnnotatedMales\tAnnotatedFemales\tMaleMale\tFemaleMale\tMaleFemale\tFemaleFemale')
for pid in sorted(pids):
	print(pid + '\t' + str(males[pid]) + '\t' + str(females[pid]) + '\t' + str(mm[pid]) + '\t' + str(fm[pid]) + '\t' + str(mf[pid]) + '\t' + str(ff[pid]))



dt = pd.read_csv(fm_obj.localEuthData, index_col=False)
dt.dissection_time = pd.to_datetime(dt.dissection_time)

label_map = {'c':'BuildEvent', 'f':'FeedEvent','p':'BuildEvent','t':'FeedEvent','b':'BuildEvent','m':'FeedEvent','s':'SpawnEvent','x':'OtherEvent','o':'OtherEvent','d':'OtherEvent'}

flag = True
for i,pid in enumerate(dt.pid):
	if pid not in ['MC_singlenuc23_1_Tk33_021220','MC_singlenuc23_8_Tk33_031720','MC_singlenuc24_4_Tk47_030320','MC_singlenuc29_3_Tk9_030320','MC_singlenuc34_3_Tk43_030320','MC_singlenuc35_11_Tk61_051220','MC_singlenuc36_2_Tk3_030320','MC_singlenuc37_2_Tk17_030320','MC_singlenuc41_2_Tk9_030920','MC_singlenuc45_7_Tk47_050720','MC_singlenuc55_2_Tk47_051220','MC_singlenuc58_4_Tk53_060220','MC_singlenuc59_4_Tk61_060220','MC_singlenuc62_3_Tk65_060220','MC_singlenuc65_4_Tk9_072920','MC_singlenuc81_1_Tk51_072920','MC_singlenuc86_b1_Tk47_073020','MC_singlenuc90_b1_Tk3_081120','MC_singlenuc91_b1_Tk9_081120',	'MC_singlenuc94_b1_Tk31_081120']:
		continue
	if pid == 'MC_singlenuc64_1_Tk51_060220':
		continue

	print(pid)

	fm_obj.setProjectID(pid)
	fm_obj.downloadData(fm_obj.localOldVideoCropFile)
	fm_obj_b.downloadData(fm_obj.localMasterDir + pid + '/MasterAnalysisFiles')
	c_dt = pd.read_csv(fm_obj.localMasterDir + pid + '/MasterAnalysisFiles/AllLabeledClusters.csv', index_col = 0)
	c_dt.TimeStamp = pd.to_datetime(c_dt.TimeStamp)

	video_crop = np.load(fm_obj.localOldVideoCropFile)
	video_crop = video_crop[:,[1,0]] # flip x and y values
	poly = Polygon(video_crop)
	c_dt['InBounds'] = [poly.contains(Point(x, y)) for x,y in zip(c_dt.X + c_dt.X_span/2, c_dt.Y + c_dt.Y_span/2)]
	c_dt['EventCategory'] = c_dt.Prediction.map(label_map)
	c_dt['TimeUntilDissection'] = (c_dt.TimeStamp - dt[dt.pid == pid].dissection_time.values[0]).dt.total_seconds()
	c_dt['TimeBins'] = c_dt.TimeUntilDissection.floordiv(60*120)*2
	c_dt = c_dt[['LID','VideoID','ClipCreated','TimeStamp','Prediction','InBounds','EventCategory','TimeUntilDissection','TimeBins']]


	t_dt = pd.read_csv(fm_obj.localMasterDir + pid + '/MasterAnalysisFiles/AllFishSex.csv')
	sex_track_dt = t_dt[t_dt.sex_p_value > 0.75].groupby('track_id').agg({'frame':'count','sex_class':'mean'}).reset_index()

	s_dt = pd.read_csv(fm_obj.localMasterDir + pid + '/MasterAnalysisFiles/AllAssociatedSex.csv', index_col = 0)
	s_dt = s_dt[['LID','VideoID','track_id']]

	all_dt = pd.merge(pd.merge(s_dt,c_dt, on = ['LID','VideoID']) ,sex_track_dt , on = 'track_id')
	all_dt['Sex'] = all_dt.sex_class.map(lambda x: 'Male' if x > 0.5 else 'Female')
	# InBound analysis
	inbounds_dt = all_dt.groupby(['InBounds','EventCategory']).count()['track_id'].reset_index()
	others = inbounds_dt[inbounds_dt.EventCategory == 'OtherEvent'].groupby('InBounds').sum()
	total = inbounds_dt.groupby('InBounds').sum()
	try:
		outbounds_other_ratio = others.loc[False]['track_id']/total.loc[False]['track_id']
		inbounds_other_ratio = others.loc[True]['track_id']/total.loc[True]['track_id']
	except:
		continue

	print('InBoundsAnalysis: OutOtherRatio = ' + str(outbounds_other_ratio) + '; InOtherRatio= ' + str(inbounds_other_ratio))
	"""
	# Time analysis
	time_dt = all_dt[all_dt.InBounds == True].groupby(['TimeBins','EventCategory']).count()['track_id']
	print('Before sacrifice: Event categories')
	try:
		print(time_dt.loc[-1.5] + time_dt.loc[-1.0] + time_dt.loc[-.5])
	except:
		try:
			print(time_dt.loc[-1.0] + time_dt.loc[-.5])
		except KeyError:
			print(time_dt.loc[-.5])
			flag = False
	print('After sacrifice: Event categories')
	try:
		print(time_dt.loc[0] + time_dt.loc[0.5])
	except KeyError:
		print(time_dt.loc[0])
	# Sex analysis
	print('Sex categories')
	print(all_dt.groupby('TimeBins').mean()['sex_class'])
#	print(np.histogram(all_dt.sex_class))
	sex_hist = np.histogram(all_dt.sex_class, bins = [0,.2,.4,.6,.8,1.0])[0]
	f_skew = (sex_hist[0] - sex_hist[-1])/(sex_hist[0] + sex_hist[-1])
	extreme_skew = (sex_hist[0] + sex_hist[-1])/sex_hist.sum()

	print(sex_hist)
	print(f_skew)
	print(extreme_skew)
	"""
	print(all_dt[(all_dt.InBounds == True) & (all_dt.frame > 30) & ((all_dt.EventCategory == 'BuildEvent') | (all_dt.EventCategory == 'FeedEvent'))].groupby(['EventCategory','Sex','TimeBins']).count()['track_id'])
	summary_dt = all_dt[(all_dt.InBounds == True) & (all_dt.frame > 30) & ((all_dt.EventCategory == 'BuildEvent') | (all_dt.EventCategory == 'FeedEvent'))].groupby(['EventCategory','Sex','TimeBins']).count()['track_id'].reset_index()
	summary_dt['ProjectID'] = pid

	try:
		all_project_data = all_project_data.append(summary_dt)
	except NameError:
		all_project_data = summary_dt

pdb.set_trace()

