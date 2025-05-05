from data_preparers.depth_preparer import DepthPreparer as DP
from helper_modules.file_manager import FileManager as FM



fm_obj = FM(projectID = 'MC_s7_tr2_BowerBuilding')
dp_obj = DP(fm_obj)
dp_obj.validateInputData()
dp_obj.createSmoothedArray()