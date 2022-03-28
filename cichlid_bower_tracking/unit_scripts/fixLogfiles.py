from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM
import shutil

fm_obj = FM()
print('collecting project IDs')
projectIDs = fm_obj.getAllProjectIDs()
print(f'{len(projectIDs)} found')
# projectIDs = ['BH_t021_MC_111521']
for i, projectID in enumerate(projectIDs):
    print(f'{i}/{len(projectIDs)}: {projectID}')
    try:
        fm_obj = FM(projectID=projectID)
        upload = True
        with open(fm_obj.localLogfile, 'r') as f:
            data = f.readlines()
            for i, line in enumerate(data):
                if line.startswith('FrameCaptured:'):
                    if ',,LOF:' in line:
                        line = line.split(',,LOF:')[0]
                    else:
                        t, = fm_obj.lp._ret_data(line, 'Time')
                        if 8 <= t.hour <= 18:
                            line = line.rstrip() + ',,LOF: True\n'
                        else:
                            line = line.rstrip() + ',,LOF: False\n'
                        data[i] = line
        with open(fm_obj.localLogfile, 'w') as f:
            f.writelines(data)
        fm_obj.uploadData(fm_obj.localLogfile)
        shutil.rmtree(fm_obj.localProjectDir)
    except Exception as e:
        print(f'project {projectID} failed with Exception: {e}')


