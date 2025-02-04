from helper_modules.file_manager import FileManager as FM
from helper_modules.log_parser import LogParser as LP
from data_preparers.prep_preparer import PrepPreparer as PrP

import numpy as np
import os, sys, pdb, subprocess, argparse
import cv2
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('tank_ID', type = str, help = 'Tank ID')
parser.add_argument('analysis_ID', type = str, help = 'Analysis ID')
parser.add_argument('project_ID', type = str, help = 'Project ID')

args = parser.parse_args()

projectID = args.project_ID
analysisID = args.analysis_ID
file_manager = FM(projectID=projectID)
# prep_preparer = PrP(fileManager=file_manager)
tankID = args.tank_ID

file_manager.downloadData(file_manager.localFirstDepthRGB)
img = cv2.imread(file_manager.localFirstDepthRGB)
pdb.set_trace()

class Crop():
    
    def __init__(self, img):
          self.img = img
          self.upload_crop_file()
          pass

    def _click_event(self, event, x, y, flags, params):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.poly) == 4:
                return
            self.poly.append((x,y))
            cv2.circle(self.interactive_pic, (x,y), 5, (255,0,0), -1)
            cv2.putText(self.interactive_pic, str(len(self.poly)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            if len(self.poly) > 1:
                for i in range(len(self.poly) - 1):
                    cv2.line(self.interactive_pic, self.poly[i], self.poly[i+1], (255,0,0), 1)
            cv2.imshow(self.interactive_text, self.interactive_pic)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.poly = []
            self.interactive_pic = self.original_pic.copy()
            cv2.imshow(self.interactive_text, self.interactive_pic)


    def upload_crop_file(self):

        
        # img = cv2.imread(file_manager.localFirstDepthRGB)
        self.poly = []
        self.original_pic = self.img
        self.interactive_pic = self.original_pic.copy()
        self.interactive_text = 'Click four points to crop. Right-click to start over. Press escape once you are finished'

        cv2.imshow(self.interactive_text, self.interactive_pic)
        cv2.setMouseCallback(self.interactive_text, self._click_event)
        cv2.waitKey(0)

        for i in range(3):
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)

        # if len(self.poly) != 4:
        #     continue
        
                    
        DepthCropFile = file_manager.localMasterDir + '__TankData/' + tankID + '/DepthCrop.txt'
        DepthCropDir = os.path.dirname(DepthCropFile)

        os.makedirs(DepthCropDir,exist_ok=True)

        with open(DepthCropFile, 'w') as f:
                print(','.join([str(x) for x in self.poly]), file = f)
                        
        file_manager.uploadData(DepthCropFile)

        mask = np.zeros_like(img, dtype=np.uint8)
        points = np.array(self.poly, dtype=np.int32).reshape((-1, 1, 2))
        # points = np.array(self.poly)
        # mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        cropped_image = cv2.bitwise_and(self.img, mask)

        pdb.set_trace()
        CroppedImg = DepthCropDir+"/MaskedImg.jpg"
        FirstDepthRGB = DepthCropDir + '/FirstDepthRGB.jpg'
        cv2.imwrite(CroppedImg, cropped_image)
        cv2.imwrite(FirstDepthRGB,self.img)


        file_manager.uploadData(CroppedImg)
        file_manager.uploadData(FirstDepthRGB)

crop_img = Crop(img)

